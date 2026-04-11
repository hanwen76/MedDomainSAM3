from __future__ import annotations

import torch
import torch.nn as nn


class LearnableMemoryPrompt(nn.Module):
    """
    A minimal free-parameter memory module for ablation.

    The memory units are directly optimized as continuous prompt tokens without
    any explicit observation encoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_tokens: int = 4,
        init_std: float = 0.02,
        prompt_scale: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_tokens = num_tokens
        self.prompt_scale = prompt_scale
        self.memory_tokens = nn.Parameter(
            torch.randn(num_tokens, hidden_dim) * init_std
        )

    def build_prompt(self, img_feats, txt_feats, txt_masks=None):
        batch_size = txt_feats.shape[1]
        prompt = self.memory_tokens.unsqueeze(1).expand(-1, batch_size, -1)
        prompt = prompt * self.prompt_scale
        prompt_mask = torch.zeros(
            (batch_size, self.num_tokens),
            device=prompt.device,
            dtype=torch.bool,
        )
        return prompt, prompt_mask
