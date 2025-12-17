import time
import torch
import torch.optim as optim
# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from sparsemax import Sparsemax

import numpy as np
from collections import defaultdict
from collections import OrderedDict
import os
from os.path import join as pjoin

from data.utils import MotionNormalizerTorch, face_joint_indx, fid_l, fid_r
from data.quaternion import *
from utils.utils import print_current_loss
from eval import evaluation_during_training
from models.mask_transformer.tools import *

from einops import rearrange, repeat

import wandb

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        self.normalizer = MotionNormalizerTorch(self.device)
        self.InteractionLoss = torch.nn.SmoothL1Loss(reduction='none')
        self.softmax = Sparsemax(dim=-1)
        
        # 랭크 0(대표 프로세스) 여부를 옵션에서 받아 둠 (train_transformer.py에서 opt.is_main을 세팅)
        self.is_main = getattr(args, "is_main", True)

        # Mixed Precision 설정
        self.use_mixed_precision = getattr(args, "use_mixed_precision", False)
        if self.use_mixed_precision:
            self.scaler = GradScaler()
            if self.is_main:
                print("Mixed Precision Training: ENABLED")
        else:
            self.scaler = None

        # Gradient Accumulation 설정
        self.gradient_accumulation_steps = getattr(args, "gradient_accumulation_steps", 1)
        self.accumulation_counter = 0
        if self.is_main and self.gradient_accumulation_steps > 1:
            print(f"Gradient Accumulation: {self.gradient_accumulation_steps} steps")

        # if args.is_train:
        #     self.logger = SummaryWriter(args.log_dir)

        # 로거는 랭크 0에서만 생성 → 다중 프로세스 중복 기록 방지
        if args.is_train and self.is_main:
            self.logger = SummaryWriter(args.log_dir)
        else:
            self.logger = None

        # Wandb 초기화 (랭크 0에서만)
        self.use_wandb = getattr(args, "use_wandb", False)
        if self.use_wandb and self.is_main:
            # Login to wandb with API key from options
            wandb_api_key = getattr(args, "wandb_api_key", "a0f95db2a87665862aefe307e54b000739821378")
            wandb.login(key=wandb_api_key)
            # wandb 초기화 - run name은 args.wandb_name 또는 args.name 사용
            wandb_run_name = getattr(args, "wandb_name", args.name)
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                name=wandb_run_name,
                config={
                    "dataset": args.dataset_name,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "max_epoch": args.max_epoch,
                    "latent_dim": args.latent_dim,
                    "ff_size": args.ff_size,
                    "n_layers": args.n_layers,
                    "n_heads": args.n_heads,
                    "dropout": args.dropout,
                    "cond_drop_prob": args.cond_drop_prob,
                    "seed": args.seed,
                    "gamma": args.gamma,
                    "step_unroll": args.step_unroll,
                    "interaction_mask_prob": args.interaction_mask_prob,
                    "distributed": getattr(args, "distributed", False),
                    "world_size": getattr(args, "world_size", 1),
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "use_mixed_precision": self.use_mixed_precision,
                }
            )
            print(f"Wandb initialized: {args.wandb_entity}/{args.wandb_project}/{args.name}")


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr
    
    
    def forward(self, batch_data):
        
        if self.opt.dataset_name == "interhuman":
            name, conds, motion1, motion2, m_lens = batch_data
        elif self.opt.dataset_name == "interx":
            _, _, conds, _, motions, m_lens, _ = batch_data
            motion1, motion2 = motions.split(6, dim=-1)

        motion1 = motion1.detach().float().to(self.device)
        motion2 = motion2.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)
        # print(f"Motions from dataset: {motion1.shape}, {motion2.shape}")
        # print(f"Motion lenghts: {m_lens}")
        
        code_idx1, _ = self.vq_model.encode(motion1)
        code_idx2, _ = self.vq_model.encode(motion2)
        code_idx = torch.cat([code_idx1, code_idx2], dim=1)
        # print(f"Code Index: {code_idx1.shape}, {code_idx2.shape}, {code_idx.shape}")
        
        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        m_lens = m_lens // 4
        # print(f"Motion Lengths: {m_lens}")

        _loss, _acc, _, _, _ = self.t2m_transformer(code_idx[..., 0], conds, m_lens)
        return _loss, _acc
        
       

    def update(self, batch_data):
        if self.use_mixed_precision:
            # Mixed Precision Training with Gradient Accumulation
            with autocast():
                loss, acc = self.forward(batch_data)

            # Scale loss for accumulation
            loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()

            self.accumulation_counter += 1

            # Update only when accumulation is complete
            if self.accumulation_counter % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.opt_t2m_transformer)
                self.scaler.update()
                self.opt_t2m_transformer.zero_grad()
                self.scheduler.step()
        else:
            # Standard Training with Gradient Accumulation
            loss, acc = self.forward(batch_data)

            # Scale loss for accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            self.accumulation_counter += 1

            # Update only when accumulation is complete
            if self.accumulation_counter % self.gradient_accumulation_steps == 0:
                self.opt_t2m_transformer.step()
                self.opt_t2m_transformer.zero_grad()
                self.scheduler.step()

        return loss.item() * self.gradient_accumulation_steps, acc

    # def save(self, file_name, ep, total_it):
    #     t2m_trans_state_dict = self.t2m_transformer.state_dict()
    #     clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_')]
    #     for e in clip_weights:
    #         del t2m_trans_state_dict[e]
    #     state = {
    #         't2m_transformer': t2m_trans_state_dict,
    #         'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
    #         'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None ,
    #         'ep': ep,
    #         'total_it': total_it,
    #     }
    #     torch.save(state, file_name)

    def save(self, file_name, ep, total_it):
        # DDP로 감쌌다면 실제 가중치는 .module 안에 있음
        model_to_save = (self.t2m_transformer.module
                        if hasattr(self.t2m_transformer, "module")
                        else self.t2m_transformer)
        t2m_trans_state_dict = model_to_save.state_dict()

        clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_')]
        for e in clip_weights:
            del t2m_trans_state_dict[e]

        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
            'ep': ep,
            'total_it': total_it,
        }

        # 랭크 0만 저장
        import torch.distributed as dist
        if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
            torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)

        # DDP로 감쌌다면 실제 가중치는 .module 안에 있음
        model_to_load = (self.t2m_transformer.module
                        if hasattr(self.t2m_transformer, "module")
                        else self.t2m_transformer)

        missing_keys, unexpected_keys = model_to_load.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        assert len(unexpected_keys) == 0
        assert all([k.startswith('clip_') for k in missing_keys])

        self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer']) # Optimizer
        try:
            self.scheduler.load_state_dict({key: checkpoint['scheduler'][key] for key in ["last_epoch", "_step_count"]}) # Scheduler
        except:
            print('Resume wo scheduler')
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, test_loader, eval_wrapper):
        if not hasattr(self.t2m_transformer, "module"):
            self.t2m_transformer.to(self.device)

        self.vq_model.to(self.device)

        # for name, p in self.t2m_transformer.named_parameters():
        #     print(name)
        
        total_iters = self.opt.max_epoch * len(train_loader)
        self.opt.milestones = [int(total_iters * 0.5), int(total_iters * 0.7), int(total_iters * 0.85)]
        self.opt.warm_up_iter = len(train_loader) // 4
        self.opt.log_every = len(train_loader) // 10
        self.opt.save_latest = len(train_loader) // 2
        
        # print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        # print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        # print(f'Milestones: {self.opt.milestones}')
        # print('Warm Up Iterations: %04d, Log Every: %04d, Save Latest: %04d' % (self.opt.warm_up_iter, self.opt.log_every, self.opt.save_latest))
        if self.is_main:
            print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
            print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
            print(f'Milestones: {self.opt.milestones}')
            print('Warm Up Iterations: %04d, Log Every: %04d, Save Latest: %04d' % (self.opt.warm_up_iter, self.opt.log_every, self.opt.save_latest))


        self.opt_t2m_transformer = optim.AdamW(self.t2m_transformer.parameters(), betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            it = it // self.opt.log_every * self.opt.log_every
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        logs = defaultdict(def_value, OrderedDict())

        max_acc = -np.inf
        min_loss = np.inf
        min_fid = np.inf
        max_top1 = -np.inf

        if self.opt.do_eval:
            eval_file = pjoin(self.opt.eval_dir, 'evaluation_training.log')

        while epoch < self.opt.max_epoch:
            epoch += 1
            self.t2m_transformer.train()
            self.vq_model.eval()

            # 추가: 분산 샘플러의 셔플 시드를 에폭마다 동기화
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            for i, batch in enumerate(train_loader):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)

                # Synchronize loss/acc across all GPUs for accurate logging
                if dist.is_available() and dist.is_initialized():
                    loss_tensor = torch.tensor([loss], device=self.device, dtype=torch.float32)
                    acc_tensor = torch.tensor([acc], device=self.device, dtype=torch.float32)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                    dist.all_reduce(acc_tensor, op=dist.ReduceOp.AVG)
                    loss = loss_tensor.item()
                    acc = acc_tensor.item()

                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']

                # if it % self.opt.log_every == 0:
                #     mean_loss = OrderedDict()
                #     # self.logger.add_scalar('val_loss', val_loss, it)
                #     # self.l
                #     for tag, value in logs.items():
                #         self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                #         mean_loss[tag] = value / self.opt.log_every
                #     logs = defaultdict(def_value, OrderedDict())
                #     print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)
                if it % self.opt.log_every == 0 and self.is_main:
                    mean_loss = OrderedDict()
                    if self.logger is not None:
                        for tag, value in logs.items():
                            self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                            mean_loss[tag] = value / self.opt.log_every

                    # Log training metrics to wandb
                    # Dictionary comprehension: iterate over accumulated logs and compute averages
                    # Example: if logs = {"loss": 93.6, "acc": 10.8} and log_every = 20
                    #          then wandb_log = {"train/loss": 4.68, "train/acc": 0.54}
                    if self.use_wandb and self.is_main:
                        wandb_log = {f"train/{tag}": value / self.opt.log_every for tag, value in logs.items()}
                        wandb_log["iteration"] = it
                        wandb_log["epoch"] = epoch
                        wandb.log(wandb_log, step=it)

                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            #print('Validation time:')
            # 출력/스칼라 기록은 랭크 0에서만
            if self.is_main:
                print('Validation time:')

            self.vq_model.eval()
            self.t2m_transformer.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            # Synchronize validation metrics across all GPUs
            if dist.is_available() and dist.is_initialized():
                # Compute local statistics
                local_loss = torch.tensor([np.mean(val_loss)], device=self.device, dtype=torch.float32)
                local_acc = torch.tensor([np.mean(val_acc)], device=self.device, dtype=torch.float32)
                local_count = torch.tensor([len(val_loss)], device=self.device, dtype=torch.float32)

                # Sum across all GPUs
                dist.all_reduce(local_loss, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_acc, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_count, op=dist.ReduceOp.SUM)

                # Compute weighted average
                global_val_loss = (local_loss / local_count).item()
                global_val_acc = (local_acc / local_count).item()
            else:
                global_val_loss = np.mean(val_loss)
                global_val_acc = np.mean(val_acc)

            #print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")
            if self.is_main:
                print(f"Validation loss:{global_val_loss:.3f}, accuracy:{global_val_acc:.3f}")

            # self.logger.add_scalar('Val/loss', np.mean(val_loss), epoch)
            # self.logger.add_scalar('Val/acc', np.mean(val_acc), epoch)
            if self.logger is not None and self.is_main:
                self.logger.add_scalar('Val/loss', global_val_loss, epoch)
                self.logger.add_scalar('Val/acc', global_val_acc, epoch)

            # Log validation metrics to wandb
            # Global metrics averaged across all GPUs for accurate monitoring
            if self.use_wandb and self.is_main:
                wandb.log({
                    "val/loss": global_val_loss,
                    "val/acc": global_val_acc,
                    "epoch": epoch
                }, step=it)

            if global_val_acc > max_acc:
                if self.is_main:
                    print(f"Improved accuracy from {max_acc:.02f} to {global_val_acc}!!!")
                self.save(pjoin(self.opt.model_dir, 'best_acc.tar'), epoch, it)
                max_acc = global_val_acc

            if global_val_loss < min_loss:
                if self.is_main:
                    print(f"Improved Loss from {min_loss:.02f} to {global_val_loss}!!!")
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_loss = global_val_loss

            if self.opt.do_eval and self.is_main:
                # eval_start_epoch부터 evaluation 시작
                eval_start = getattr(self.opt, 'eval_start_epoch', 0)
                if epoch >= eval_start and epoch % self.opt.eval_every_e == 0:
                    self.vq_model.eval()
                    self.t2m_transformer.eval()
                    # DDP 모델의 경우 .module로 접근
                    eval_trans = (self.t2m_transformer.module
                                if hasattr(self.t2m_transformer, 'module')
                                else self.t2m_transformer)
                    fid, mat, top1 = evaluation_during_training(self.opt, self.vq_model, test_loader,
                                                                eval_wrapper, epoch, eval_file, trans=eval_trans)
                   
                    # self.logger.add_scalar('Test/FID', fid, epoch)
                    # self.logger.add_scalar('Test/Matching', mat, epoch)
                    # self.logger.add_scalar('Test/Top1', top1, epoch)
                    if self.logger is not None:
                        self.logger.add_scalar('Test/FID', fid, epoch)
                        self.logger.add_scalar('Test/Matching', mat, epoch)
                        self.logger.add_scalar('Test/Top1', top1, epoch)

                    # Log evaluation metrics to wandb
                    # FID: Frechet Inception Distance (lower is better)
                    # mat: Matching score, top1: Top-1 accuracy (higher is better)
                    if self.use_wandb and self.is_main:
                        wandb.log({
                            "eval/fid": fid,
                            "eval/matching": mat,
                            "eval/top1": top1,
                            "epoch": epoch
                        }, step=it)
                        
                    if fid < min_fid:
                        min_fid = fid
                        self.save(pjoin(self.opt.model_dir, 'best_fid.tar'), epoch, it)
                        print('Best FID Model So Far!~')

                    if top1 > max_top1:
                        max_top1 = top1
                        self.save(pjoin(self.opt.model_dir, 'best_top1.tar'), epoch, it)
                        print('Best Top1 Model So Far!~')

            print('\n')

        # Finish wandb run when training is complete
        # wandb.finish() uploads any remaining data and closes the run
        if self.use_wandb and self.is_main:
            wandb.finish()