from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.medical_memory_bank import StaticMedicalMemoryBank


class ImageMemoryPromptBuilder(nn.Module):
    """
    Build prompt tokens for the SAM3 image branch.

    The builder supports two modes:
    - "task_encoder": encode the entire support bank into compact task tokens
      using learnable queries and support-conditioned attention.
    - "retrieval": keep the original nearest-neighbor memory prompt behavior.

    In both cases, the output is a prompt token sequence that can be
    concatenated with text and geometry prompts.
    """

    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int = 64,
        topk: int = 4,
        image_weight: float = 1.0,
        text_weight: float = 1.0,
        prompt_scale: float = 1.0,
        prompt_mode: str = "task_encoder",
        num_task_tokens: int = 4,
        task_temperature: float = 1.0,
        bank_path: Optional[str] = None,
    ):
        super().__init__()
        self.topk = topk
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.prompt_scale = prompt_scale
        self.prompt_mode = prompt_mode
        self.num_task_tokens = num_task_tokens
        self.task_temperature = task_temperature
        self.bank = StaticMedicalMemoryBank(bank_path=bank_path, topk=topk)
        image_key_dim = memory_dim
        if self.bank.is_loaded() and self.bank.memory_keys is not None:
            image_key_dim = self.bank.memory_keys.shape[1]
        text_key_dim = image_key_dim
        if self.bank.is_loaded() and self.bank.memory_text_keys is not None:
            text_key_dim = self.bank.memory_text_keys.shape[1]
        value_dim = image_key_dim
        if self.bank.is_loaded() and self.bank.memory_features is not None:
            value_dim = self.bank.memory_features.shape[1]
        task_value_dim = hidden_dim
        if self.bank.is_loaded() and getattr(self.bank, "task_embeddings", None) is not None:
            task_value_dim = self.bank.task_embeddings.shape[-1]
        self.image_query_proj = nn.Linear(hidden_dim, image_key_dim)
        self.text_query_proj = nn.Linear(hidden_dim, text_key_dim)
        self.query_context_proj = nn.Linear(image_key_dim + text_key_dim, hidden_dim)
        self.memory_prompt_proj = nn.Linear(image_key_dim, hidden_dim)
        if value_dim != image_key_dim:
            self.memory_prompt_proj = nn.Linear(value_dim, hidden_dim)
        self.task_bank_proj = nn.Linear(task_value_dim, hidden_dim)
        self.task_query_tokens = nn.Parameter(
            torch.randn(num_task_tokens, hidden_dim) * 0.02
        )
        self.task_cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=8, batch_first=True
        )
        self.task_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.task_norm1 = nn.LayerNorm(hidden_dim)
        self.task_norm2 = nn.LayerNorm(hidden_dim)

    def is_enabled(self) -> bool:
        return (
            self.bank is not None
            and (
                self.bank.is_loaded()
                or (
                    getattr(self.bank, "task_embeddings", None) is not None
                    and self.bank.memory_keys is not None
                )
            )
        )

    def set_task_support(
        self,
        task_embeddings: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_text_keys: torch.Tensor | None = None,
    ):
        """Inject sampled support-pool task tokens without using an offline bank."""
        target_device = task_embeddings.device
        self.bank.task_embeddings = task_embeddings.detach().float().contiguous()
        self.bank.memory_keys = memory_keys.detach().float().contiguous()
        self.bank.memory_text_keys = (
            memory_text_keys.detach().float().contiguous()
            if memory_text_keys is not None
            else None
        )
        self.bank.memory_features = None
        self.bank.memory_pos_enc = None
        self._refresh_projection_shapes(target_device)

    def set_memory(
        self,
        memory_features: torch.Tensor,
        memory_pos_enc: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_text_keys: torch.Tensor | None = None,
        task_embeddings: torch.Tensor | None = None,
    ):
        """Inject a memory bank directly without requiring a file path."""
        target_device = memory_features.device
        self.bank.memory_features = memory_features.detach().float().contiguous()
        self.bank.memory_pos_enc = memory_pos_enc.detach().float().contiguous()
        self.bank.memory_keys = memory_keys.detach().float().contiguous()
        self.bank.memory_text_keys = (
            memory_text_keys.detach().float().contiguous() if memory_text_keys is not None else None
        )
        self.bank.task_embeddings = (
            task_embeddings.detach().float().contiguous() if task_embeddings is not None else None
        )
        self._refresh_projection_shapes(target_device)

    def _refresh_projection_shapes(self, target_device):
        # Update projection dimensions if needed
        if self.bank.memory_keys is not None:
            image_key_dim = self.bank.memory_keys.shape[1]
            text_key_dim = image_key_dim
            if self.bank.memory_text_keys is not None:
                text_key_dim = self.bank.memory_text_keys.shape[1]
            value_dim = image_key_dim
            if self.bank.memory_features is not None:
                value_dim = self.bank.memory_features.shape[1]
            task_value_dim = self.task_bank_proj.in_features
            if self.bank.task_embeddings is not None:
                task_value_dim = self.bank.task_embeddings.shape[-1]
            # Rebuild projection layers if dimension changed
            if (
                self.image_query_proj.out_features != image_key_dim
                or self.text_query_proj.out_features != text_key_dim
                or self.query_context_proj.in_features != image_key_dim + text_key_dim
                or self.memory_prompt_proj.in_features != value_dim
                or self.task_bank_proj.in_features != task_value_dim
            ):
                self.image_query_proj = nn.Linear(self.image_query_proj.in_features, image_key_dim)
                self.text_query_proj = nn.Linear(self.text_query_proj.in_features, text_key_dim)
                self.query_context_proj = nn.Linear(
                    image_key_dim + text_key_dim, self.query_context_proj.out_features
                )
                self.memory_prompt_proj = nn.Linear(
                    value_dim, self.memory_prompt_proj.out_features
                )
                self.task_bank_proj = nn.Linear(
                    task_value_dim, self.task_bank_proj.out_features
                )
        # Ensure all projection layers are on the same device as the memory bank
        self.image_query_proj = self.image_query_proj.to(target_device)
        self.text_query_proj = self.text_query_proj.to(target_device)
        self.query_context_proj = self.query_context_proj.to(target_device)
        self.memory_prompt_proj = self.memory_prompt_proj.to(target_device)
        self.task_bank_proj = self.task_bank_proj.to(target_device)
        if self.task_query_tokens.device != target_device:
            self.task_query_tokens = nn.Parameter(
                self.task_query_tokens.detach().to(target_device)
            )
        self.task_cross_attn = self.task_cross_attn.to(target_device)
        self.task_ffn = self.task_ffn.to(target_device)
        self.task_norm1 = self.task_norm1.to(target_device)
        self.task_norm2 = self.task_norm2.to(target_device)

    def _masked_text_pool(
        self,
        txt_feats: torch.Tensor,
        txt_masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # txt_feats: [T, B, C]
        txt_feats_b = txt_feats.permute(1, 0, 2)
        if txt_masks is None:
            return txt_feats_b.mean(dim=1)
        valid = (~txt_masks.bool()).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        return (txt_feats_b * valid).sum(dim=1) / denom

    def build_prompt(
        self,
        img_feats: list[torch.Tensor],
        txt_feats: torch.Tensor,
        txt_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_enabled():
            device = txt_feats.device
            batch_size = txt_feats.shape[1]
            empty_prompt = torch.zeros((0, batch_size, txt_feats.shape[-1]), device=device)
            empty_mask = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
            return empty_prompt, empty_mask

        if self.prompt_mode == "retrieval":
            return self._build_retrieval_prompt(img_feats, txt_feats, txt_masks)
        return self._build_task_encoder_prompt(img_feats, txt_feats, txt_masks)

    def _build_retrieval_prompt(
        self,
        img_feats: list[torch.Tensor],
        txt_feats: torch.Tensor,
        txt_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        top_img_feat = img_feats[-1]  # [HW, B, C]
        image_query = top_img_feat.mean(dim=0)
        image_query = self.image_query_proj(image_query)

        text_query = self._masked_text_pool(txt_feats, txt_masks)
        text_query = self.text_query_proj(text_query)

        query = image_query.detach().float().cpu()
        text_query_cpu = None
        if self.bank.memory_text_keys is not None:
            text_query_cpu = text_query.detach().float().cpu()

        scores, indices = self.bank.retriever(
            query,
            self.bank.memory_keys.cpu(),
            topk=self.topk,
            text_query=text_query_cpu,
            memory_text_keys=self.bank.memory_text_keys.cpu() if self.bank.memory_text_keys is not None else None,
            image_weight=self.image_weight,
            text_weight=self.text_weight,
        )

        all_prompts = []
        for batch_idx in range(query.shape[0]):
            batch_indices = indices[batch_idx]
            batch_scores = scores[batch_idx]
            memory_feats = self.bank.memory_features[batch_indices].to(txt_feats.device)
            pooled_memory = F.adaptive_avg_pool2d(memory_feats, output_size=1).flatten(1)
            prompt_tokens = self.memory_prompt_proj(pooled_memory)
            token_weights = torch.softmax(batch_scores.to(prompt_tokens.device), dim=0)
            prompt_tokens = prompt_tokens * token_weights.unsqueeze(-1) * self.prompt_scale
            all_prompts.append(prompt_tokens)

        memory_prompt = torch.stack(all_prompts, dim=1)  # [K, B, C]
        memory_mask = torch.zeros(
            (memory_prompt.shape[1], memory_prompt.shape[0]),
            device=memory_prompt.device,
            dtype=torch.bool,
        )
        return memory_prompt, memory_mask

    def _build_task_encoder_prompt(
        self,
        img_feats: list[torch.Tensor],
        txt_feats: torch.Tensor,
        txt_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = txt_feats.device
        batch_size = txt_feats.shape[1]

        if getattr(self.bank, "task_embeddings", None) is not None:
            support_bank = self.bank.task_embeddings.to(device)
            if support_bank.shape[-1] != txt_feats.shape[-1]:
                support_bank = self.task_bank_proj(support_bank)
        else:
            memory_features = self.bank.memory_features.to(device)
            pooled_memory = F.adaptive_avg_pool2d(memory_features, output_size=1).flatten(1)
            support_bank = self.memory_prompt_proj(pooled_memory).unsqueeze(1)  # [N, 1, C]

        top_img_feat = img_feats[-1]  # [HW, B, C]
        pooled_image_query = top_img_feat.mean(dim=0)
        if self.bank.memory_keys.shape[1] == pooled_image_query.shape[-1]:
            image_query = pooled_image_query
        else:
            image_query = self.image_query_proj(pooled_image_query)
        text_query = self._masked_text_pool(txt_feats, txt_masks)

        image_scores = torch.matmul(
            F.normalize(image_query, dim=-1),
            F.normalize(self.bank.memory_keys.to(device), dim=-1).T,
        )
        if self.bank.memory_text_keys is not None:
            if self.bank.memory_text_keys.shape[1] != text_query.shape[-1]:
                text_query = self.text_query_proj(text_query)
            text_scores = torch.matmul(text_query, self.bank.memory_text_keys.to(device).T)
            scores = self.image_weight * image_scores + self.text_weight * text_scores
        else:
            scores = image_scores
        scores = scores / max(self.task_temperature, 1e-6)
        weights = torch.softmax(scores, dim=-1)  # [B, N]

        support_tokens = torch.einsum("bn,ntc->btc", weights, support_bank)
        support_summary = support_tokens.mean(dim=1)
        query_context = self.query_context_proj(torch.cat([image_query, text_query], dim=-1))
        task_queries = self.task_query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        task_queries = task_queries + query_context.unsqueeze(1) + support_summary.unsqueeze(1)
        task_attn, _ = self.task_cross_attn(task_queries, support_tokens, support_tokens)
        task_tokens = self.task_norm1(task_queries + task_attn)
        task_tokens = self.task_norm2(task_tokens + self.task_ffn(task_tokens))
        task_tokens = task_tokens * self.prompt_scale

        memory_prompt = task_tokens.permute(1, 0, 2).contiguous()  # [K, B, C]
        memory_mask = torch.zeros(
            (memory_prompt.shape[1], memory_prompt.shape[0]),
            device=memory_prompt.device,
            dtype=torch.bool,
        )
        return memory_prompt, memory_mask


class CombinedMemoryPromptBuilder(nn.Module):
    """
    Combine a task encoder prompt builder with a learnable prompt tuner.

    This keeps the existing prompt-stream interface unchanged while allowing
    support-conditioned task tokens and free prompt embeddings to coexist.
    """

    def __init__(
        self,
        task_builder: nn.Module | None = None,
        prompt_tuning_builder: nn.Module | None = None,
    ):
        super().__init__()
        self.task_builder = task_builder
        self.prompt_tuning_builder = prompt_tuning_builder

    def is_enabled(self) -> bool:
        task_enabled = (
            self.task_builder is not None
            and hasattr(self.task_builder, "is_enabled")
            and self.task_builder.is_enabled()
        )
        tuning_enabled = self.prompt_tuning_builder is not None
        return task_enabled or tuning_enabled

    def build_prompt(
        self,
        img_feats: list[torch.Tensor],
        txt_feats: torch.Tensor,
        txt_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_chunks = []
        mask_chunks = []

        if self.task_builder is not None:
            task_prompt, task_mask = self.task_builder.build_prompt(
                img_feats=img_feats,
                txt_feats=txt_feats,
                txt_masks=txt_masks,
            )
            if task_prompt.numel() > 0:
                prompt_chunks.append(task_prompt)
                mask_chunks.append(task_mask)

        if self.prompt_tuning_builder is not None:
            tuning_prompt, tuning_mask = self.prompt_tuning_builder.build_prompt(
                img_feats=img_feats,
                txt_feats=txt_feats,
                txt_masks=txt_masks,
            )
            if tuning_prompt.numel() > 0:
                prompt_chunks.append(tuning_prompt)
                mask_chunks.append(tuning_mask)

        if not prompt_chunks:
            batch_size = txt_feats.shape[1]
            device = txt_feats.device
            empty_prompt = torch.zeros((0, batch_size, txt_feats.shape[-1]), device=device)
            empty_mask = torch.zeros((batch_size, 0), device=device, dtype=torch.bool)
            return empty_prompt, empty_mask

        if len(prompt_chunks) == 1:
            return prompt_chunks[0], mask_chunks[0]

        prompt = torch.cat(prompt_chunks, dim=0)
        mask = torch.cat(mask_chunks, dim=1)
        return prompt, mask
