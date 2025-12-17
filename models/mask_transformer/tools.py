import torch
import torch.nn.functional as F
import math
from einops import rearrange
import torch.distributed as dist
from utils.ddp_random import ddp_aware_bernoulli

# return mask where padding is FALSE
def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).repeat(len(lengths), 2) < lengths.unsqueeze(1)
    # mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

# return mask where padding is ALL FALSE
def get_pad_mask_idx(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1)

# Given seq: (b, s)
# Return mat: (1, s, s)
# Example Output:
#        [[[ True, False, False],
#          [ True,  True, False],
#          [ True,  True,  True]]]
# For causal attention
def get_subsequent_mask(seq):
    sz_b, seq_len = seq.shape
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, seq_len, seq_len)), diagonal=1)).bool()
    return subsequent_mask.to(seq.device)


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# tensor helpers

# Get a random subset of TRUE mask, with prob
def get_mask_subset_prob(mask, prob):
    # DDP-aware Bernoulli 샘플링
    # Multi-GPU 환경에서 Single GPU와 동일한 난수 시퀀스를 보장
    # ddp_aware_bernoulli: 전체 배치(batch_size * world_size) 생성 후 rank별로 분할
    # Single GPU: (batch, seq_len) 전체 생성
    # Multi-GPU: 각 GPU가 (batch * world_size, seq_len) 생성 후 자신의 (batch, seq_len) 부분만 사용
    bernoulli_mask = ddp_aware_bernoulli(prob, mask.shape, device=mask.device).bool()
    return bernoulli_mask & mask


# Get mask of special_tokens in ids
def get_mask_special_tokens(ids, special_ids):
    mask = torch.zeros_like(ids).bool()
    for special_id in special_ids:
        mask |= (ids==special_id)
    return mask

# network builder helpers
def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

# classifier free guidance functions

def uniform(shape, device=None, min=0, max=1):
    """
    Uniform random number generation with DDP-aware support.

    In DDP mode, ensures all GPUs generate the full batch, then each GPU
    selects its rank-specific portion, maintaining exact reproducibility
    with Single GPU training.

    Args:
        shape: int or tuple, local batch size (e.g., 13)
        device: torch device
        min, max: range for uniform distribution

    Returns:
        Tensor of shape `shape` with uniform random values

    Example:
        Single GPU: uniform(52) → full 52 values
        Multi-GPU:
          All GPUs generate 52 values (same seed)
          GPU 0: selects [0:13]
          GPU 1: selects [13:26]
          GPU 2: selects [26:39]
          GPU 3: selects [39:52]
    """

    # Check if DDP is active
    if dist.is_available() and dist.is_initialized():
        # DDP-aware version
        if isinstance(shape, int):
            shape = (shape,)

        local_batch_size = shape[0]
        other_dims = shape[1:] if len(shape) > 1 else ()

        # Calculate total batch size across all GPUs
        world_size = dist.get_world_size()
        total_batch_size = local_batch_size * world_size

        # Generate full batch (all GPUs generate the same sequence)
        full_size = (total_batch_size,) + other_dims
        full_uniform = torch.zeros(full_size, device=device).float().uniform_(min, max)

        # Select rank-specific portion
        rank = dist.get_rank()
        start_idx = rank * local_batch_size
        end_idx = start_idx + local_batch_size
        local_uniform = full_uniform[start_idx:end_idx]

        return local_uniform
    else:
        # Single GPU version (original behavior)
        return torch.zeros(shape, device=device).float().uniform_(min, max)

def prob_mask_like(shape, prob, device=None):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return uniform(shape, device=device) < prob

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = 1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


# Example input:
#        [[ 0.3596,  0.0862,  0.9771, -1.0000, -1.0000, -1.0000],
#         [ 0.4141,  0.1781,  0.6628,  0.5721, -1.0000, -1.0000],
#         [ 0.9428,  0.3586,  0.1659,  0.8172,  0.9273, -1.0000]]
# Example output:
#        [[  -inf,   -inf, 0.9771,   -inf,   -inf,   -inf],
#         [  -inf,   -inf, 0.6628,   -inf,   -inf,   -inf],
#         [0.9428,   -inf,   -inf,   -inf,   -inf,   -inf]]
def top_k(logits, thres = 0.9, dim = 1):
    k = math.ceil((1 - thres) * logits.shape[dim])
    val, ind = logits.topk(k, dim = dim)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(dim, ind, val)
    # func verified
    # print(probs)
    # print(logits)
    # raise
    return probs

# noise schedules

# More on large value, less on small
def cosine_schedule(t):
    return torch.cos(t * math.pi * 0.5)
def cosine_schedule_backward(x):
    return torch.acos(x) / (math.pi * 0.5)

def scale_cosine_schedule(t, scale):
    return torch.clip(scale*torch.cos(t * math.pi * 0.5) + 1 - scale, min=0., max=1.)

# More on small value, less on large
def q_schedule(bs, low, high, device):
    noise = uniform((bs,), device=device)
    schedule = 1 - cosine_schedule(noise)
    return torch.round(schedule * (high - low - 1)).long() + low

def cal_performance(pred, labels, ignore_index=None, smoothing=0., tk=1):
    loss = cal_loss(pred, labels, ignore_index, smoothing=smoothing)
    # pred_id = torch.argmax(pred, dim=1)
    # mask = labels.ne(ignore_index)
    # n_correct = pred_id.eq(labels).masked_select(mask)
    # acc = torch.mean(n_correct.float()).item()
    pred_id_k = torch.topk(pred, k=tk, dim=1).indices
    pred_id = pred_id_k[:, 0]
    mask = labels.ne(ignore_index)
    n_correct = (pred_id_k == labels.unsqueeze(1)).any(dim=1).masked_select(mask)
    acc = torch.mean(n_correct.float()).item()

    return loss, pred_id, acc


def cal_loss(pred, labels, ignore_index=None, smoothing=0.):
    '''Calculate cross entropy loss, apply label smoothing if needed.'''
    # print(pred.shape, labels.shape) #torch.Size([64, 1028, 55]) torch.Size([64, 55])
    # print(pred.shape, labels.shape) #torch.Size([64, 1027, 55]) torch.Size([64, 55])
    if smoothing:
        space = 2
        n_class = pred.size(1)
        mask = labels.ne(ignore_index)
        one_hot = rearrange(F.one_hot(labels, n_class + space), 'a ... b -> a b ...')[:, :n_class]
        # one_hot = torch.zeros_like(pred).scatter(1, labels.unsqueeze(1), 1)
        sm_one_hot = one_hot * (1 - smoothing) + (1 - one_hot) * smoothing / (n_class - 1)
        neg_log_prb = -F.log_softmax(pred, dim=1)
        loss = (sm_one_hot * neg_log_prb).sum(dim=1)
        # loss = F.cross_entropy(pred, sm_one_hot, reduction='none')
        loss = torch.mean(loss.masked_select(mask))
    else:
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            raise ValueError("NaN/Inf detected in logits before CE")

        if ignore_index is not None:
            valid = labels != ignore_index
            if valid.any():
                max_lab = labels[valid].max().item()
                min_lab = labels[valid].min().item()
                if max_lab >= pred.size(1) or min_lab < 0:
                    raise ValueError(
                        f"Label out of range: min={min_lab}, max={max_lab}, "
                        f"C={pred.size(1)}, ignore_index={ignore_index}"
                    )
        else:
            if labels.numel() > 0:
                max_lab = labels.max().item()
                min_lab = labels.min().item()
                if max_lab >= pred.size(1) or min_lab < 0:
                    raise ValueError(
                        f"Label out of range: min={min_lab}, max={max_lab}, C={pred.size(1)}"
                    )
                
        loss = F.cross_entropy(pred, labels, ignore_index=ignore_index)

    return loss