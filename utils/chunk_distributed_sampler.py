"""
ChunkDistributedSampler: Sequential Chunk-based Data Distribution

Purpose:
  DDP-aware Random과 호환되는 청크 방식 데이터 분할

  기본 DistributedSampler (인터리빙):
    GPU 0: [0, 4, 8, 12, ...]
    GPU 1: [1, 5, 9, 13, ...]

  ChunkDistributedSampler (청크):
    GPU 0: [0, 1, 2, ..., 12]
    GPU 1: [13, 14, ..., 25]

Key Difference from torch.utils.data.distributed.DistributedSampler:
  - Line 119 in original: indices[self.rank:self.total_size:self.num_replicas]  (interleaving)
  - This implementation: indices[start_idx:end_idx]                              (chunking)
"""

import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist

__all__ = ["ChunkDistributedSampler"]

T_co = TypeVar('T_co', covariant=True)


class ChunkDistributedSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset using CHUNKING.

    Unlike the default DistributedSampler which uses interleaving (rank 0 gets indices
    [0, 4, 8, ...]), this sampler uses chunking (rank 0 gets indices [0, 1, 2, ...]).

    This is essential for DDP-aware Random compatibility, where:
      - All GPUs generate the full batch random tensor (e.g., size 52)
      - Each GPU selects its rank-specific chunk (e.g., GPU 0: [0:13], GPU 1: [13:26])

    With chunking, data indices match random tensor indices perfectly:
      - GPU 0: data indices [0-12] → random indices [0-12] ✅
      - GPU 1: data indices [13-25] → random indices [13-25] ✅

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training.
        rank (int, optional): Rank of the current process within num_replicas.
        shuffle (bool, optional): If True, sampler will shuffle the indices.
        seed (int, optional): Random seed used to shuffle the sampler if shuffle=True.
        drop_last (bool, optional): If True, drop the tail to make evenly divisible.

    Example::
        >>> sampler = ChunkDistributedSampler(dataset, shuffle=True, drop_last=True)
        >>> loader = DataLoader(dataset, batch_size=13, sampler=sampler)
        >>> for epoch in range(n_epochs):
        ...     sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Calculate samples per rank
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Drop tail to make evenly divisible
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[T_co]:
        # Step 1: Shuffle (deterministic based on epoch and seed)
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Step 2: Padding or truncation
        if not self.drop_last:
            # Add extra samples to make evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail to make evenly divisible
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size, \
            f"Expected {self.total_size} indices, got {len(indices)}"

        # Step 3: CHUNKING (핵심 차이점)
        # 기본 DistributedSampler: indices[self.rank:self.total_size:self.num_replicas]  (인터리빙)
        # ChunkDistributedSampler: 연속된 청크 선택
        chunk_size = self.total_size // self.num_replicas
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size

        indices = indices[start_idx:end_idx]

        assert len(indices) == self.num_samples, \
            f"Rank {self.rank}: Expected {self.num_samples} samples, got {len(indices)}"

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When shuffle=True, this ensures all replicas
        use a different random ordering for each epoch.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
