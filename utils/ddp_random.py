"""
DDP-aware Random Number Generator for Reproducibility

Purpose:
  Single GPU와 Multi-GPU가 동일한 난수 시퀀스를 사용하도록 보장
  배치 사이즈에 관계없이 동작 (예: batch=13×4=52, batch=9×6=54 등)

Key Idea:
  모든 GPU가 동일한 seed로 전체 배치(local_batch × world_size)의 난수를 생성한 후,
  각 GPU가 자신의 rank에 해당하는 부분만 선택

Example (4 GPUs, local_batch=13, total=52):
  Single GPU:
    torch.manual_seed(seed)
    rand = torch.rand(52, 750)  # 전체 사용

  Multi-GPU (4 GPUs):
    모든 GPU: torch.manual_seed(seed)
    full_rand = torch.rand(52, 750)  # 전체 생성
    GPU 0: rand = full_rand[0:13]    # rank=0 몫
    GPU 1: rand = full_rand[13:26]   # rank=1 몫
    GPU 2: rand = full_rand[26:39]   # rank=2 몫
    GPU 3: rand = full_rand[39:52]   # rank=3 몫

Example (6 GPUs, local_batch=9, total=54):
  Single GPU:
    rand = torch.rand(54, 750)  # 전체 사용

  Multi-GPU (6 GPUs):
    full_rand = torch.rand(54, 750)
    GPU 0: rand = full_rand[0:9]
    GPU 1: rand = full_rand[9:18]
    ...
    GPU 5: rand = full_rand[45:54]
"""

import torch
import torch.distributed as dist


def get_effective_batch_size(local_batch_size):
    """
    DDP 환경에서 전체 배치 크기 계산

    Args:
        local_batch_size: 현재 GPU의 배치 크기 (예: 13 for 4 GPUs, 9 for 6 GPUs)

    Returns:
        effective_batch_size: 전체 배치 크기 (예: 13×4=52, 9×6=54)
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        return local_batch_size * world_size
    else:
        return local_batch_size


def get_rank_slice(total_size, local_size):
    """
    현재 rank에 해당하는 slice 인덱스 계산

    Args:
        total_size: 전체 배치 크기 (예: 52)
        local_size: 현재 GPU 배치 크기 (예: 13)

    Returns:
        (start_idx, end_idx): 현재 rank의 슬라이스 (예: rank=0이면 (0, 13))
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        start_idx = rank * local_size
        end_idx = start_idx + local_size
        return start_idx, end_idx
    else:
        return 0, local_size


def ddp_aware_rand(size, device, generator=None):
    """
    DDP-aware torch.rand() 대체 함수

    Single GPU와 동일한 난수 시퀀스를 보장하면서,
    Multi-GPU 환경에서는 각 GPU가 자신의 rank 몫만 받음

    Args:
        size: (batch_size, *other_dims) - local batch size 기준
        device: torch device
        generator: optional torch.Generator

    Returns:
        rand_tensor: 난수 텐서 (local batch size)

    Example:
        # Single GPU (batch=52)
        rand = ddp_aware_rand((52, 750), device)  # shape: (52, 750)

        # Multi-GPU (batch=13 per GPU, 4 GPUs)
        rand = ddp_aware_rand((13, 750), device)  # shape: (13, 750)
        # 하지만 내부적으로는 (52, 750)을 생성하고 rank별로 분할
    """
    local_batch_size = size[0]
    other_dims = size[1:]

    # 전체 배치 크기 계산
    total_batch_size = get_effective_batch_size(local_batch_size)

    # 전체 배치에 대한 난수 생성 (모든 GPU가 동일한 난수 생성)
    full_size = (total_batch_size,) + other_dims
    full_rand = torch.rand(full_size, device=device, generator=generator)

    # 현재 rank에 해당하는 부분만 추출
    start_idx, end_idx = get_rank_slice(total_batch_size, local_batch_size)
    local_rand = full_rand[start_idx:end_idx]

    return local_rand


def ddp_aware_randint(low, high, size, device, dtype=torch.int64, generator=None):
    """
    DDP-aware torch.randint() 대체 함수

    Args:
        low: 최소값
        high: 최대값 (exclusive)
        size: (batch_size, *other_dims)
        device: torch device
        dtype: data type
        generator: optional torch.Generator

    Returns:
        randint_tensor: 정수 난수 텐서 (local batch size)
    """
    local_batch_size = size[0]
    other_dims = size[1:]

    total_batch_size = get_effective_batch_size(local_batch_size)

    full_size = (total_batch_size,) + other_dims
    full_randint = torch.randint(low, high, full_size, device=device, dtype=dtype, generator=generator)

    start_idx, end_idx = get_rank_slice(total_batch_size, local_batch_size)
    local_randint = full_randint[start_idx:end_idx]

    return local_randint


def ddp_aware_bernoulli(p, size, device, generator=None):
    """
    DDP-aware Bernoulli sampling

    Args:
        p: Probability (0-1)
        size: (batch_size,) or (batch_size, *other_dims)
        device: torch device
        generator: optional torch.Generator

    Returns:
        bernoulli_tensor: Boolean tensor (local batch size)
    """
    if isinstance(size, int):
        size = (size,)

    local_batch_size = size[0]
    other_dims = size[1:] if len(size) > 1 else ()

    total_batch_size = get_effective_batch_size(local_batch_size)

    full_size = (total_batch_size,) + other_dims
    full_probs = torch.ones(full_size, device=device) * p
    full_bernoulli = torch.bernoulli(full_probs, generator=generator)

    start_idx, end_idx = get_rank_slice(total_batch_size, local_batch_size)
    local_bernoulli = full_bernoulli[start_idx:end_idx]

    return local_bernoulli


def ddp_aware_randn(size, device, generator=None):
    """
    DDP-aware torch.randn() replacement (normal distribution)

    Guarantees identical random sequence with Single GPU while
    each GPU receives only its rank-specific portion in Multi-GPU.

    Args:
        size: (batch_size, *other_dims) - local batch size basis
        device: torch device
        generator: optional torch.Generator

    Returns:
        randn_tensor: Normal distribution random tensor (local batch size)

    Example:
        # Single GPU (batch=256)
        randn = ddp_aware_randn((256, 512), device)  # shape: (256, 512)

        # Multi-GPU (batch=42 per GPU, 6 GPUs)
        randn = ddp_aware_randn((42, 512), device)  # shape: (42, 512)
        # Internally generates (252, 512) and splits by rank
    """
    local_batch_size = size[0]
    other_dims = size[1:]

    # Calculate total batch size
    total_batch_size = get_effective_batch_size(local_batch_size)

    # Generate random numbers for full batch (all GPUs generate identical tensor)
    full_size = (total_batch_size,) + other_dims
    full_randn = torch.randn(full_size, device=device, generator=generator)

    # Extract portion for current rank
    start_idx, end_idx = get_rank_slice(total_batch_size, local_batch_size)
    local_randn = full_randn[start_idx:end_idx]

    return local_randn


# 편의 함수: torch.rand 스타일 호출 지원
def rand_like_single_gpu(local_batch_size, *other_dims, device='cuda', generator=None):
    """
    사용 편의를 위한 wrapper

    Example:
        # transformer.py에서
        # 기존: torch.rand((bs, ntokens), device=device)
        # 변경: rand_like_single_gpu(bs, ntokens, device=device)

        rand = rand_like_single_gpu(13, 750, device='cuda')  # shape: (13, 750)
        # Multi-GPU에서도 Single GPU의 seed=0 시퀀스와 동일
    """
    size = (local_batch_size,) + other_dims
    return ddp_aware_rand(size, device, generator)


def test_ddp_aware_random():
    """
    테스트 함수: Single GPU vs Multi-GPU 난수 일치 확인
    """
    import numpy as np

    print("=== Testing DDP-aware Random ===")

    # Simulate Single GPU
    torch.manual_seed(0)
    single_gpu_rand = torch.rand(52, 10, device='cpu')
    print(f"Single GPU (batch=52):")
    print(f"  First sample:  {single_gpu_rand[0, :5]}")
    print(f"  Sample 13:     {single_gpu_rand[13, :5]}")
    print(f"  Sample 26:     {single_gpu_rand[26, :5]}")
    print(f"  Last sample:   {single_gpu_rand[51, :5]}")

    # Simulate Multi-GPU (without actual DDP initialization)
    # Manually set world_size and rank
    print(f"\nSimulated Multi-GPU (4 GPUs, batch=13 each):")

    for rank in range(4):
        torch.manual_seed(0)  # 모든 GPU가 동일한 seed

        # 전체 생성
        full_rand = torch.rand(52, 10, device='cpu')

        # Rank별 분할
        start = rank * 13
        end = start + 13
        local_rand = full_rand[start:end]

        print(f"  GPU {rank} (rank={rank}):")
        print(f"    First local sample: {local_rand[0, :5]}")
        print(f"    Matches Single GPU sample {start}: {torch.allclose(local_rand[0], single_gpu_rand[start])}")

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_ddp_aware_random()
