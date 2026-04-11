from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from sam3.model.medical_memory_retriever import MedicalMemoryRetriever


@dataclass
class MedicalMemoryBatch:
    memory_features: torch.Tensor
    memory_pos_enc: torch.Tensor
    memory_keys: torch.Tensor
    memory_text_keys: torch.Tensor | None
    scores: torch.Tensor
    indices: torch.Tensor


class StaticMedicalMemoryBank:
    """
    Lightweight bank for offline-built case memories.

    Expected checkpoint format:
    {
        "memory_features": [N, C_mem, H, W],
        "memory_pos_enc": [N, C_mem, H, W],
        "memory_keys": [N, C_mem],  # optional
        "metadata": ...             # optional
    }
    """

    def __init__(self, bank_path: Optional[str] = None, topk: int = 4):
        self.bank_path = bank_path
        self.retriever = MedicalMemoryRetriever(topk=topk)
        self.memory_features = None
        self.memory_pos_enc = None
        self.memory_keys = None
        self.memory_text_keys = None
        self.metadata = None
        if bank_path is not None:
            self.load(bank_path)

    def is_loaded(self) -> bool:
        return self.memory_features is not None and self.memory_pos_enc is not None

    def load(self, bank_path: Optional[str] = None):
        path = bank_path or self.bank_path
        if path is None:
            raise ValueError("bank_path must be provided to load static memories")
        payload = torch.load(path, map_location="cpu")
        self.memory_features = payload["memory_features"].float().contiguous()
        self.memory_pos_enc = payload["memory_pos_enc"].float().contiguous()
        self.memory_keys = payload.get("memory_keys")
        if self.memory_keys is None:
            self.memory_keys = self._pool_memory_keys(self.memory_features)
        else:
            self.memory_keys = self.memory_keys.float().contiguous()
        self.memory_text_keys = payload.get("memory_text_keys")
        if self.memory_text_keys is not None:
            self.memory_text_keys = self.memory_text_keys.float().contiguous()
        self.metadata = payload.get("metadata")
        self.bank_path = path
        return self

    def _pool_memory_keys(self, memory_features: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(memory_features, output_size=1)
        return pooled.flatten(1)

    def retrieve(
        self,
        query: torch.Tensor,
        topk: Optional[int] = None,
        text_query: Optional[torch.Tensor] = None,
        image_weight: float = 1.0,
        text_weight: float = 1.0,
    ) -> MedicalMemoryBatch:
        if not self.is_loaded():
            raise RuntimeError("Static medical memory bank has not been loaded")

        query_cpu = query.detach().float().cpu()
        text_query_cpu = None
        if text_query is not None and self.memory_text_keys is not None:
            text_query_cpu = text_query.detach().float().cpu()
        scores, indices = self.retriever(
            query_cpu,
            self.memory_keys,
            topk=topk,
            text_query=text_query_cpu,
            memory_text_keys=self.memory_text_keys,
            image_weight=image_weight,
            text_weight=text_weight,
        )
        indices = indices[0]
        scores = scores[0]
        return MedicalMemoryBatch(
            memory_features=self.memory_features[indices],
            memory_pos_enc=self.memory_pos_enc[indices],
            memory_keys=self.memory_keys[indices],
            memory_text_keys=(
                None
                if self.memory_text_keys is None
                else self.memory_text_keys[indices]
            ),
            scores=scores,
            indices=indices,
        )
