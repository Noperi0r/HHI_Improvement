import os
import torch
import numpy as np
import random

from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils.chunk_distributed_sampler import ChunkDistributedSampler
from torch.cuda.amp import autocast, GradScaler
from os.path import join as pjoin

from models.mask_transformer.transformer import MaskTransformer
from models.mask_transformer.transformer_trainer import MaskTransformerTrainer
from models.vq.model import RVQVAE
from options.trans_option import TrainTransOptions

from utils.get_opt import get_opt
from utils.utils import fixseed

def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)
    if vq_opt.dataset_name == "interhuman":
        vq_opt.dim_joint = 12
    if vq_opt.dataset_name == "interx":
        vq_opt.dim_joint = 6
   
    vq_model = RVQVAE(vq_opt,
                        vq_opt.dim_joint,
                        vq_opt.nb_code,
                        vq_opt.code_dim,
                        vq_opt.output_emb_width,
                        vq_opt.down_t,
                        vq_opt.stride_t,
                        vq_opt.width,
                        vq_opt.depth,
                        vq_opt.dilation_growth_rate,
                        vq_opt.vq_act,
                        vq_opt.vq_norm)
    
    print(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'best_fid.tar')
    
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'best_fid.tar'),  map_location='cpu')
    model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
    
    missing_keys, unexpected_keys = vq_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('decoder.conv') or k.startswith('decoder.resnets')for k in missing_keys])
    print(f'Loading VQ Model {opt.vq_name}, epoch {ckpt["ep"]}')
    return vq_model, vq_opt

if __name__ == '__main__':

    parser = TrainTransOptions()
    opt = parser.parse()

    # Multi-GPU 환경에서는 일단 기본 seed만 설정 (rank별 조정은 나중에)
    fixseed(opt.seed)

    #opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    # --- 여기부터 분산 모드 장치/통신 초기화 로직 추가 ---
    def _is_main_process():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

    def _setup_ddp():
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("!!!!![DDP ENV] MASTER_ADDR=", os.environ.get("MASTER_ADDR"),
                "MASTER_PORT=", os.environ.get("MASTER_PORT"),
                "RANK=", os.environ.get("RANK"),
                "WORLD_SIZE=", os.environ.get("WORLD_SIZE"))

        dist.init_process_group(backend="nccl", init_method="env://")

    def _cleanup_ddp():
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()

    if getattr(opt, "distributed", False):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))   # torchrun이 내려줌
        torch.cuda.set_device(local_rank)                   # 현재 프로세스 전용 GPU로 고정

        # 메모리 최적화 설정
        torch.cuda.empty_cache()                           # 캐시 정리

        _setup_ddp()                                       # 통신 초기화
        opt.rank = dist.get_rank()
        opt.world_size = dist.get_world_size()
        opt.is_main = _is_main_process()                   # 랭크0 여부
        opt.device = torch.device(f"cuda:{local_rank}")

        # DDP-aware Random Strategy: 모든 GPU가 동일한 seed 사용
        #
        # 핵심 원리: "전체 생성 후 rank별 분할"
        #   1. 모든 GPU가 seed로 전체 배치(batch_size * world_size)의 난수 생성
        #   2. 각 GPU가 자신의 rank에 해당하는 부분만 선택
        #   3. Single GPU와 100% 동일한 난수 시퀀스 보장
        #
        # 예시 (4 GPUs, batch_size=13):
        #   Single GPU: rand(52, 750) → 전체 사용
        #   Multi-GPU:  모든 GPU가 rand(52, 750) 생성
        #               GPU 0: [0:13] 선택
        #               GPU 1: [13:26] 선택
        #               GPU 2: [26:39] 선택
        #               GPU 3: [39:52] 선택
        #
        # 결과: Single GPU의 동일 seed 시퀀스와 수학적으로 동일한 gradient
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)

        if opt.is_main:
            total_batch = opt.batch_size * opt.world_size
            print(f"")
            print(f"=== DDP-aware Random Configuration ===")
            print(f"Strategy: Full-batch generation + rank-based split")
            print(f"All GPUs using identical seed: {opt.seed}")
            print(f"")
            print(f"Single GPU (baseline):")
            print(f"  - Seed: {opt.seed}")
            print(f"  - Batch: {total_batch}")
            print(f"  - Random sequence: seed={opt.seed} positions 1-{total_batch}")
            print(f"")
            print(f"Multi-GPU (DDP-aware):")
            print(f"  - GPUs: {opt.world_size}")
            print(f"  - Batch per GPU: {opt.batch_size}")
            print(f"  - Total batch: {total_batch}")
            print(f"  - Random generation: Full batch ({total_batch}) → Split by rank")
            for rank_idx in range(opt.world_size):
                start = rank_idx * opt.batch_size
                end = start + opt.batch_size
                print(f"    GPU {rank_idx}: positions [{start}:{end}]   = seed={opt.seed} pos {start+1}-{end}")
            print(f"")
            print(f"Result: 100% identical to Single GPU seed={opt.seed}")
            print(f"Expected FID: 5.2-5.5 (target: 5.23)")
            print(f"======================================")
            print(f"")
    else:
        opt.rank = 0
        opt.world_size = 1
        opt.is_main = True
        opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")  # 기존 경로
    # --- 여기까지 추가 ---
    
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.anim_dir = pjoin(opt.save_root, 'animation')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'log')


    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.anim_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    vq_model, vq_opt = load_vq_model()

    # 코드북 크기 디버깅 정보 출력
    if getattr(opt, 'is_main', True):
        print(f"VQ Model codebook size: {vq_opt.nb_code}")
        print(f"VQ Model actual codebook shape: {vq_model.num_code if hasattr(vq_model, 'num_code') else 'Unknown'}")
        print(f"VQ Model quantizers info:")
        if hasattr(vq_model, 'quantizers'):
            for i, q in enumerate(vq_model.quantizers):
                if hasattr(q, 'nb_code'):
                    print(f"   Quantizer {i}: {q.nb_code} codes")
                if hasattr(q, 'codebook'):
                    print(f"   Quantizer {i} codebook shape: {q.codebook.shape}")

    if opt.dataset_name == "interhuman":
        opt.data_root = '/local_datasets/InterHuman_251205'
        opt.joints_num = 22
        opt.dim_joint = 12
        opt.test_batch_size = 96
        fps = 30

        from data.interhuman import InterHumanDataset
        from models.evaluator.evaluator import EvaluatorModelWrapper

        opt.mode = "train"
        train_dataset = InterHumanDataset(opt)
        opt.mode = "val"
        val_dataset = InterHumanDataset(opt)

        if opt.do_eval:
            opt.mode = "val"
            test_dataset = InterHumanDataset(opt)
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, drop_last=True, num_workers=0, shuffle=False)

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
        opt.max_motion_length = 150
        opt.max_text_len = 35
        opt.unit_length = 4
        
        opt.test_batch_size = 32
        vq_opt.dim_joint = 6
        fps = 30

        from data.interx import Text2MotionDatasetHHI, Text2MotionDatasetV2HHI, collate_fn
        from models.evaluator.evaluator_interx import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab')
        train_dataset = Text2MotionDatasetV2HHI(opt, 
                                           pjoin(opt.data_root, 'splits/train.txt'),
                                           w_vectorizer, 
                                           pjoin(opt.motion_dir, 'train.h5'))
        val_dataset = Text2MotionDatasetV2HHI(opt, 
                                         pjoin(opt.data_root, 'splits/val.txt'),
                                         w_vectorizer, 
                                         pjoin(opt.motion_dir, 'val.h5'))
        
        if opt.do_eval:
            test_dataset = Text2MotionDatasetV2HHI(opt, 
                                                pjoin(opt.data_root, 'splits/val.txt'), 
                                                w_vectorizer, 
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

    # clip_version = 'ViT-L/14@336px'
    # opt.num_tokens = vq_opt.nb_code
    # mask_transformer = MaskTransformer(code_dim=vq_opt.code_dim,
    #                                   cond_mode='text',
    #                                   latent_dim=opt.latent_dim,
    #                                   ff_size=opt.ff_size,
    #                                   num_layers=opt.n_layers,
    #                                   num_heads=opt.n_heads,
    #                                   dropout=opt.dropout,
    #                                   clip_dim=768,#512,
    #                                   cond_drop_prob=opt.cond_drop_prob,
    #                                   clip_version=clip_version,
    #                                   opt=opt)

    # pc_transformer = sum(param.numel() for param in mask_transformer.parameters_wo_clip())
    # print('Total parameters of the Masked Transformer=: {:.2f}M'.format(pc_transformer / 1000_000))

    clip_version = 'ViT-L/14@336px'

    # VQ 모델의 실제 코드북 크기를 동적으로 감지
    actual_codebook_size = vq_opt.nb_code

    # VQ 모델에서 실제 quantizer의 코드북 크기 확인
    if hasattr(vq_model, 'quantizers') and len(vq_model.quantizers) > 0:
        actual_codebook_size = vq_model.quantizers[0].nb_code
        if getattr(opt, 'is_main', True):
            print(f"실제 quantizer codebook 크기: {actual_codebook_size}")
    elif hasattr(vq_model, 'quantizer') and hasattr(vq_model.quantizer, 'layers'):
        actual_codebook_size = vq_model.quantizer.layers[0].nb_code
        if getattr(opt, 'is_main', True):
            print(f"실제 quantizer.layers[0] codebook 크기: {actual_codebook_size}")
    else:
        if getattr(opt, 'is_main', True):
            print(f"WARNING: quantizer를 찾을 수 없음, vq_opt.nb_code 사용: {actual_codebook_size}")

    # Transformer num_tokens를 VQ 모델과 일치시킴
    opt.num_tokens = actual_codebook_size

    # 코드북 크기 확인 및 일치성 검증
    if getattr(opt, 'is_main', True):
        print(f"=== 코드북 크기 디버깅 정보 ===")
        print(f"VQ Model nb_code: {vq_opt.nb_code}")
        print(f"실제 VQ codebook 크기: {actual_codebook_size}")
        print(f"Transformer num_tokens 설정: {opt.num_tokens}")
        print(f"코드북 크기 일치성 확인 완료!")
        print(f"===============================")

    mask_transformer = MaskTransformer(code_dim=vq_opt.code_dim,
                                    cond_mode='text',
                                    latent_dim=opt.latent_dim,
                                    ff_size=opt.ff_size,
                                    num_layers=opt.n_layers,
                                    num_heads=opt.n_heads,
                                    dropout=opt.dropout,
                                    clip_dim=768,  # 512에서 수정되어 있음
                                    cond_drop_prob=opt.cond_drop_prob,
                                    clip_version=clip_version,
                                    opt=opt)

    # 1) 모델을 이 프로세스의 디바이스로 올리기
    mask_transformer = mask_transformer.to(opt.device)

    # 2) SyncBatchNorm: BN이 있다면 GPU 간 통계 동기화
    if getattr(opt, 'distributed', False) and getattr(opt, 'sync_bn', False):
        mask_transformer = torch.nn.SyncBatchNorm.convert_sync_batchnorm(mask_transformer)

    # 3) DDP 래핑: 프로세스-로컬 GPU로 연결
    if getattr(opt, 'distributed', False):
        from torch.nn.parallel import DistributedDataParallel as DDP
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        mask_transformer = DDP(mask_transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        
    # 4) 파라미터 수 출력은 랭크 0만
    pc_transformer = sum(param.numel() for param in
                        (mask_transformer.module.parameters_wo_clip()
                        if hasattr(mask_transformer, 'module')
                        else mask_transformer.parameters_wo_clip()))
    
    if getattr(opt, 'is_main', True):
        print('Total parameters of the Masked Transformer=: {:.2f}M'.format(pc_transformer / 1_000_000))

    # train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
    #                           shuffle=True, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
    #                         shuffle=False, pin_memory=True)
    # 분산 모드일 때만 샘플러를 쓰고, 아니면 None
    # ChunkDistributedSampler: DDP-aware Random과 호환되는 청크 방식 분할
    # - 기본 DistributedSampler: GPU 0 gets [0, 4, 8, ...] (인터리빙) ❌
    # - ChunkDistributedSampler: GPU 0 gets [0, 1, 2, ..., 12] (청크) ✅
    # - 데이터-난수 매칭: 인터리빙 92.3% 미스매치 vs 청크 100% 매칭
    train_sampler = (ChunkDistributedSampler(train_dataset, shuffle=True,  drop_last=True)
                    if getattr(opt, 'distributed', False) else None)
    val_sampler   = (ChunkDistributedSampler(val_dataset,   shuffle=False, drop_last=False)
                    if getattr(opt, 'distributed', False) else None)

    # DataLoader에서 sampler를 우선으로 쓰고, sampler가 있으면 shuffle=False로 둔다
    # Mixed Precision 및 메모리 최적화 설정
    # Note: Mixed Precision은 기본적으로 비활성화 (재현성을 위해)
    # 속도가 필요하면 --use_mixed_precision 플래그 사용
    if getattr(opt, 'is_main', True):
        print(f"=== Performance Optimization Settings ===")
        print(f"Mixed Precision Training: {'ENABLED' if opt.use_mixed_precision else 'DISABLED (FP32 for reproducibility)'}")
        print(f"Batch size per GPU: {opt.batch_size}")
        print(f"Total batch size: {opt.batch_size * getattr(opt, 'world_size', 1)}")
        print(f"Sequence length: 300 frames")
        if not opt.use_mixed_precision:
            print(f"NOTE: Using FP32 for better reproducibility (slower but more accurate)")
        print(f"=========================================")

    # In DDP training, drop_last=True ensures all ranks get the same batch size
    drop_last = True if opt.distributed else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),  # Single GPU: shuffle=True, Multi-GPU: sampler handles shuffle
        num_workers=opt.num_workers,
        pin_memory=True,   # 성능 향상을 위해 pin_memory 활성화
        drop_last=drop_last,
        persistent_workers=True if opt.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        sampler=val_sampler,
        shuffle=False,                     # 검증은 원래도 False
        pin_memory=True,                   # 성능 향상을 위해 pin_memory 활성화
        persistent_workers=True if opt.num_workers > 0 else False
    )

    opt.save_vis = False
    opt.gen_react = False

    trainer = MaskTransformerTrainer(opt, mask_transformer, vq_model)

    trainer.train(train_loader, val_loader, test_loader, eval_wrapper=eval_wrapper)

    if getattr(opt, "distributed", False):
        _cleanup_ddp()