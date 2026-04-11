import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalMemoryRetriever(nn.Module):
    """Retrieve the closest memory entries with cosine similarity."""

    def __init__(self, topk=4):
        super().__init__()
        self.topk = topk

    def forward(
        self,
        query,
        memory_keys,
        topk=None,
        text_query=None,
        memory_text_keys=None,
        image_weight=1.0,
        text_weight=1.0,
    ):
        if memory_keys.numel() == 0:
            raise ValueError("memory_keys is empty")

        if query.dim() == 1:
            query = query.unsqueeze(0)

        query = F.normalize(query, dim=-1)
        memory_keys = F.normalize(memory_keys, dim=-1)
        similarity = image_weight * torch.matmul(query, memory_keys.transpose(0, 1))
        if text_query is not None and memory_text_keys is not None:
            text_query = F.normalize(text_query, dim=-1)
            memory_text_keys = F.normalize(memory_text_keys, dim=-1)
            similarity = similarity + text_weight * torch.matmul(
                text_query, memory_text_keys.transpose(0, 1)
            )
        k = min(topk or self.topk, memory_keys.size(0))
        scores, indices = torch.topk(similarity, k=k, dim=-1)
        return scores, indices
