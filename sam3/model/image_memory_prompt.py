from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.medical_memory_bank import StaticMedicalMemoryBank


class ImageMemoryPromptBuilder(nn.Module):
    """
    Build lightweight memory prompt tokens for the SAM3 image branch.

    The prompt tokens are retrieved from an offline memory bank using image/text
    similarity and then projected to the SAM3 hidden dimension so they can be
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
        bank_path: Optional[str] = None,
    ):
        super().__init__()
        self.topk = topk
        self.image_weight = image_weight
        self.text_weight = text_weight
        self.prompt_scale = prompt_scale
        self.bank = StaticMedicalMemoryBank(bank_path=bank_path, topk=topk)
        image_key_dim = memory_dim
        if self.bank.is_loaded() and self.bank.memory_keys is not None:
            image_key_dim = self.bank.memory_keys.shape[1]
        text_key_dim = image_key_dim
        if self.bank.is_loaded() and self.bank.memory_text_keys is not None:
            text_key_dim = self.bank.memory_text_keys.shape[1]
        self.image_query_proj = nn.Linear(hidden_dim, image_key_dim)
        self.text_query_proj = nn.Linear(hidden_dim, text_key_dim)
        self.memory_prompt_proj = nn.Linear(image_key_dim, hidden_dim)

    def is_enabled(self) -> bool:
        return self.bank is not None and self.bank.is_loaded()

    def set_memory(
        self,
        memory_features: torch.Tensor,
        memory_pos_enc: torch.Tensor,
        memory_keys: torch.Tensor,
        memory_text_keys: torch.Tensor | None = None,
    ):
        """Inject a memory bank directly without requiring a file path."""
        target_device = memory_features.device
        self.bank.memory_features = memory_features.detach().float().contiguous()
        self.bank.memory_pos_enc = memory_pos_enc.detach().float().contiguous()
        self.bank.memory_keys = memory_keys.detach().float().contiguous()
        self.bank.memory_text_keys = (
            memory_text_keys.detach().float().contiguous() if memory_text_keys is not None else None
        )
        # Update projection dimensions if needed
        if self.bank.memory_keys is not None:
            image_key_dim = self.bank.memory_keys.shape[1]
            text_key_dim = image_key_dim
            if self.bank.memory_text_keys is not None:
                text_key_dim = self.bank.memory_text_keys.shape[1]
            # Rebuild projection layers if dimension changed
            if self.image_query_proj.out_features != image_key_dim:
                self.image_query_proj = nn.Linear(self.image_query_proj.in_features, image_key_dim)
                self.text_query_proj = nn.Linear(self.text_query_proj.in_features, text_key_dim)
                self.memory_prompt_proj = nn.Linear(image_key_dim, self.memory_prompt_proj.out_features)
        # Ensure all projection layers are on the same device as the memory bank
        self.image_query_proj = self.image_query_proj.to(target_device)
        self.text_query_proj = self.text_query_proj.to(target_device)
        self.memory_prompt_proj = self.memory_prompt_proj.to(target_device)

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
