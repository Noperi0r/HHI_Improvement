import os
from os.path import join as pjoin
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils.chunk_distributed_sampler import ChunkDistributedSampler

from models.vq.model import RVQVAE
from models.vq.vq_trainer import RVQTokenizerTrainer
from options.vq_option import arg_parse


from utils.get_opt import get_opt

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    opt = arg_parse(True)

    # --- DDP initialization functions ---
    def _is_main_process():
        """Check if current process is rank 0 (for logging/saving)"""
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def _setup_ddp():
        """Initialize distributed communication group"""
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("!!!!![DDP ENV] MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
                "MASTER_PORT=", os.environ.get("MASTER_PORT"),
                "RANK=", os.environ.get("RANK"),
                "WORLD_SIZE=", os.environ.get("WORLD_SIZE"))

        # NCCL: NVIDIA's library for GPU-to-GPU communication
        dist.init_process_group(backend="nccl", init_method="env://")

    def _cleanup_ddp():
        """Cleanup DDP"""
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    # --- Device setup for DDP mode ---
    if getattr(opt, "distributed", False):
        # Get local rank from environment variable set by torchrun
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)  # Bind this process to specific GPU

        torch.cuda.empty_cache()  # Clear GPU memory cache

        _setup_ddp()  # Initialize communication group
        opt.rank = dist.get_rank()  # Global rank (0 ~ world_size-1)
        opt.world_size = dist.get_world_size()  # Total number of processes
        opt.is_main = _is_main_process()  # Whether this is rank 0
        opt.device = torch.device(f"cuda:{local_rank}")

        # All GPUs use identical seed (KEY for reproducibility!)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

        if opt.is_main:
            total_batch = opt.batch_size * opt.world_size
            print(f"")
            print(f"=== DDP Configuration ===")
            print(f"World Size: {opt.world_size}")
            print(f"Rank: {opt.rank}")
            print(f"Batch per GPU: {opt.batch_size}")
            print(f"Total Batch: {total_batch}")
            print(f"Seed (all GPUs): {opt.seed}")
            print(f"========================")
            print(f"")
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.is_main = True
        opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")

        # Single GPU also needs seed setting
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.anim_dir = pjoin(opt.save_root, 'animation')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.anim_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    if opt.dataset_name == "interhuman":
        opt.data_root = '/local_datasets/InterHuman'
        opt.joints_num = 22
        opt.dim_joint = 12
        opt.test_batch_size = 96
        fps = 30
        
        # lazy import
        from data.interhuman import InterHumanMotion, InterHumanDataset
        from models.evaluator.evaluator import EvaluatorModelWrapper

        opt.mode = "train"
        train_dataset = InterHumanMotion(opt)
        opt.mode = "val"
        val_dataset = InterHumanMotion(opt)

        if opt.do_eval:
            opt.mode = "val"
            test_dataset = InterHumanDataset(opt)
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, 
                                    drop_last=True, num_workers=0, shuffle=False)

            evalmodel_cfg = get_opt("checkpoints/eval_model/eval_model.yaml", opt.device, complete=False)
            eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, opt.device)
        else:
            test_loader = None
            eval_wrapper = None
    
    elif opt.dataset_name == "interx":
        opt.data_root = '/local_datasets/InterX'
        opt.motion_dir = pjoin(opt.data_root, 'processed/motions')
        opt.text_dir = pjoin(opt.data_root, 'processed/texts_processed')
        opt.motion_rep = "smpl"
        opt.joints_num = 56 
        opt.dim_joint = 6
        opt.max_motion_length = 150
        opt.max_text_len = 35
        opt.unit_length = 4
        
        opt.test_batch_size = 32
        fps = 30

        from data.interx import MotionDatasetV2HHI, Text2MotionDatasetV2HHI, collate_fn
        from models.evaluator.evaluator_interx import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        train_dataset = MotionDatasetV2HHI(opt, 
                                           pjoin(opt.data_root, 'splits/train.txt'), 
                                           pjoin(opt.motion_dir, 'train.h5'))
        val_dataset = MotionDatasetV2HHI(opt, 
                                         pjoin(opt.data_root, 'splits/val.txt'), 
                                         pjoin(opt.motion_dir, 'val.h5'))
        
        if opt.do_eval:
            test_dataset = Text2MotionDatasetV2HHI(opt, 
                                                pjoin(opt.data_root, 'splits/val.txt'), 
                                                WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab'), 
                                                pjoin(opt.motion_dir, 'val.h5'))
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, 
                                    num_workers=4, drop_last=True, collate_fn=collate_fn, shuffle=True)
            
            wrapper_opt = get_opt("checkpoints/hhi/Comp_v6_KLD01/opt.txt", opt.device, complete=False)
            eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        else:
            test_loader = None
            eval_wrapper = None

    else:
        raise KeyError('Dataset Does not Exists')
    
    
    net = RVQVAE(opt,
                opt.dim_joint,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    # Move model to device BEFORE DDP wrapping
    net = net.to(opt.device)

    # SyncBatchNorm: synchronize BatchNorm statistics across GPUs (optional)
    if getattr(opt, 'distributed', False) and getattr(opt, 'sync_bn', False):
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # DDP wrapping: create model replica on each GPU
    if getattr(opt, 'distributed', False):
        from torch.nn.parallel import DistributedDataParallel as DDP
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        net = DDP(
            net,
            device_ids=[local_rank],  # Current GPU
            output_device=local_rank,  # Output to current GPU
            find_unused_parameters=True  # VQVAE uses all parameters
        )

    # Calculate parameters (handle DDP wrapping with .module)
    model_for_count = net.module if hasattr(net, 'module') else net
    pc_vq = sum(param.numel() for param in model_for_count.parameters())
    pc_vq_enc = sum(param.numel() for param in model_for_count.encoder.parameters())
    pc_vq_dec = sum(param.numel() for param in model_for_count.decoder.parameters())

    if getattr(opt, 'is_main', True):
        print(net if not hasattr(net, 'module') else net.module)
        print('Total parameters of VQVAE: {}M'.format(pc_vq/1000_000))
        print('Total parameters of encoder: {}M'.format(pc_vq_enc/1000_000))
        print('Total parameters of decoder: {}M'.format(pc_vq_dec/1000_000))
        print('Total parameters of all models: {}M'.format((pc_vq_enc+pc_vq_dec)/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    # ChunkDistributedSampler: DDP-aware Random compatible data splitting
    # - Default DistributedSampler: GPU 0 gets [0, 4, 8, ...] (interleaving) ❌
    # - ChunkDistributedSampler: GPU 0 gets [0, 1, 2, ..., batch_size-1] (chunking) ✅
    # - Data-random matching: interleaving 92.3% mismatch vs chunking 100% match
    train_sampler = (ChunkDistributedSampler(train_dataset, shuffle=True, drop_last=True)
                    if getattr(opt, 'distributed', False) else None)
    val_sampler = (ChunkDistributedSampler(val_dataset, shuffle=False, drop_last=False)
                  if getattr(opt, 'distributed', False) else None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,  # Use ChunkSampler in distributed mode
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True if opt.distributed else False,
        shuffle=False if train_sampler else True,  # Sampler handles shuffling
        persistent_workers=True if opt.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        sampler=val_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,  # Validation never shuffles
        persistent_workers=True if opt.num_workers > 0 else False
    )

    opt.save_vis = False
    trainer.train(train_loader, val_loader, test_loader=test_loader, eval_wrapper=eval_wrapper)

    # Cleanup DDP
    if getattr(opt, "distributed", False):
        _cleanup_ddp()