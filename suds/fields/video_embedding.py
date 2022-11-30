from typing import Any

import suds_cuda
import torch
from torch import nn


class VideoEmbeddingFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, times: torch.Tensor, video_ids: torch.Tensor, weights: torch.Tensor,
                num_frequencies: int) -> torch.Tensor:
        embeddings = suds_cuda.video_embedding_forward(times, video_ids, weights, num_frequencies)
        ctx.save_for_backward(times, video_ids, torch.IntTensor([weights.shape[0], num_frequencies]))
        return embeddings

    @staticmethod
    def backward(ctx: Any, d_loss_embedding: torch.Tensor):
        times, video_ids, num_sequences_and_frequencies = ctx.saved_tensors
        d_loss_weights = suds_cuda.video_embedding_backward(d_loss_embedding.contiguous(), times, video_ids,
                                                            num_sequences_and_frequencies[0].item(),
                                                            num_sequences_and_frequencies[1].item())

        return None, None, d_loss_weights, None


class VideoEmbedding(nn.Module):

    def __init__(self, num_videos: int, num_frequencies: int, embedding_dim: int):
        super(VideoEmbedding, self).__init__()

        self.num_frequencies = num_frequencies
        self.sequence_code_weights = nn.Parameter(
            torch.empty(size=(num_videos, embedding_dim, num_frequencies * 2 + 1), dtype=torch.float32),
            requires_grad=True)
        torch.nn.init.normal_(self.sequence_code_weights)

    def forward(self, times: torch.Tensor, video_ids: torch.Tensor) -> torch.Tensor:
        return VideoEmbeddingFunction.apply(times, video_ids, self.sequence_code_weights, self.num_frequencies)
