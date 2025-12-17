import torch
import torch.nn as nn
import numpy as np
# from networks.layers import *
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from models.mask_transformer.transformer_modules import *
from models.mask_transformer.tools import *
from torch.distributions.categorical import Categorical
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D
from utils.ddp_random import ddp_aware_rand, ddp_aware_randint, ddp_aware_bernoulli

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2)) # [seqen, bs, input_feats]
        # print(x.shape)
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class SpaTempPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(SpaTempPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.positional_encoding = PositionalEncoding2D(d_model)
    
    def forward(self, x):
        seqlen, bs, input_feats = x.shape
        x1, sep, x2  = x.split([seqlen//2, 1, seqlen//2])
        
        def add_positional_encoding(x):
            x = x.permute(1,0,2) # [seqen, bs, input_feats] -> [bs, seqen, input_feats]
            x = x.reshape(x.shape[0], 5, x.shape[1]//5, x.shape[2])

            x = x + self.positional_encoding(x)

            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])
            x = x.permute(1,0,2)

            return x

        x1 = add_positional_encoding(x1)
        x2 = add_positional_encoding(x2)
        
        x = torch.cat([x1, sep, x2], dim=0)
        return self.dropout(x)

class OutputProcess_adaLN(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        
        self.LayerNorm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_mod = AdaLNModulation(latent_dim, nchunks=2)
        
        self.poseFinal = nn.Linear(latent_dim, out_feats) #Bias!

    def forward(self, hidden_states: torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        
        shift, scale = self.adaLN_mod(cond)
        hidden_states = modulate(self.LayerNorm(hidden_states), shift, scale)

        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


class MaskTransformer(nn.Module):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1, 
                 clip_version=None, opt=None, **kargs):
        super(MaskTransformer, self).__init__()
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.nbp = 5
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)
        '''
        Preparing Networks
        '''
        self.input_process = InputProcess(self.code_dim, self.latent_dim)

        self.position_enc = SpaTempPositionalEncoding(self.latent_dim, self.dropout)

        self.Transformer = InterMTransformer(d_model=self.latent_dim,
                                        nhead=num_heads,
                                        dim_feedforward=ff_size,
                                        dropout=dropout,
                                        num_layers=num_layers,
                                        nbp=self.nbp)

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # if self.cond_mode != 'no_cond':
        if self.cond_mode == 'text':
            self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.cond_emb = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")


        _num_tokens = opt.num_tokens + 3  # three dummy tokens, one for masking, one for padding, one for separating
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1
        self.sep_id = opt.num_tokens + 2

        self.output_process = OutputProcess_adaLN(out_feats=opt.num_tokens, latent_dim=latent_dim)

        # 디버깅: output_process의 실제 출력 크기 확인
        print(f"[DEBUG] OutputProcess_adaLN out_feats 설정: {opt.num_tokens}")
        print(f"[DEBUG] OutputProcess poseFinal weight shape: {self.output_process.poseFinal.weight.shape}")
        print(f"[DEBUG] VQ codebook size: {opt.num_tokens}, Total tokens with specials: {_num_tokens}")
        print(f"[DEBUG] Special tokens - mask_id: {opt.num_tokens}, pad_id: {opt.num_tokens + 1}, sep_id: {opt.num_tokens + 2}")

        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self.initialize_weights()
        
        '''
        Preparing frozen weights
        '''

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.load_and_freeze_clip(clip_version)

        self.noise_schedule = cosine_schedule
        self.noise_schedule_backward = cosine_schedule_backward

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        :return:
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0)) #add two dummy tokens, 0 vectors
        self.token_emb.requires_grad_(False)
        # self.token_emb.weight.requires_grad = False
        # self.token_emb_ready = True
        print("Token embedding initialized!")

    def initialize_weights(self):
        def __init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    module.bias.data.zero_()
                if module.weight is not None:
                    module.weight.data.fill_(1.0)
        
        self.apply(__init_weights)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.Transformer.blocks:
            nn.init.constant_(block.adaLN_mod_combined.model[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod_combined.model[-1].bias, 0)
            nn.init.constant_(block.adaLN_mod_split.model[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod_split.model[-1].bias, 0)
            
        
        # nn.init.normal_(self.position_enc.pe, mean=0.0, std=0.02)
        nn.init.constant_(self.output_process.adaLN_mod.model[-1].weight, 0)
        nn.init.constant_(self.output_process.adaLN_mod.model[-1].bias, 0)

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_')]

    def load_and_freeze_clip(self, clip_version):

        ##From InterGen
        clip_model, _ = clip.load(clip_version, device="cpu", jit=False)

        self.clip_token_embedding = clip_model.token_embedding
        self.clip_transformer = clip_model.transformer
        self.clip_positional_embedding = clip_model.positional_embedding
        self.clip_ln_final = clip_model.ln_final
        self.clip_dtype = clip_model.dtype

        for p in self.clip_transformer.parameters():
            p.requires_grad = False
        for p in self.clip_token_embedding.parameters():
            p.requires_grad = False
        for p in self.clip_ln_final.parameters():
            p.requires_grad = False
        
        clipTransLayer = nn.TransformerEncoderLayer(d_model=768,
                                                    nhead=8,
                                                    dim_feedforward=2048,
                                                    dropout=0.1,
                                                    activation="gelu",
                                                    batch_first=True)
        self.clipTrans = nn.TransformerEncoder(clipTransLayer, num_layers=2)
        self.clipln = nn.LayerNorm(768)

        # NEW: Word-level projection for cross-attention
        # Project CLIP's 768 dim to InterMask's latent_dim (d_model of transformer)
        # CRITICAL: Use latent_dim (transformer d_model), NOT clip_dim (sentence embedding dim)
        self.word_proj = nn.Linear(768, self.latent_dim)

    def encode_text(self, raw_text):
        device = next(self.parameters()).device

        # From InterGen
        with torch.no_grad():
            text = clip.tokenize(raw_text, truncate=True).to(device)
            x = self.clip_token_embedding(text).type(self.clip_dtype)
            pe_tokens = x + self.clip_positional_embedding.type(self.clip_dtype)
            x = pe_tokens.permute(1,0,2)
            x = self.clip_transformer(x)
            x = x.permute(1,0,2)
            clip_out = self.clip_ln_final(x).type(self.clip_dtype)

        out = self.clipTrans(clip_out)
        out = self.clipln(out)

        # Extract sentence-level embedding (EOS token position)
        # Used for global conditioning via AdaLN-mod
        feat_clip_sent = out[torch.arange(out.shape[0]), text.argmax(dim=-1)]  # [B, 768]

        # Extract word-level embeddings (all token positions)
        # Used for word-motion cross-attention
        feat_clip_words = self.word_proj(out)  # [B, M, 768] -> [B, M, latent_dim]

        # Create word mask (True for valid words, False for padding)
        # CLIP uses 0 for padding tokens
        word_mask = (text != 0)  # [B, M]

        return feat_clip_sent, feat_clip_words, word_mask

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            # DDP-aware: Single GPU와 동일한 난수 시퀀스 보장
            mask = ddp_aware_bernoulli(self.cond_drop_prob, (bs,), device=cond.device).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def compute_alignment_losses(self, word_mask=None):
        """
        Compute alignment losses from attention maps stored in transformer blocks.

        **MODIFIED**: Coverage loss removed to prevent treating all words equally.
        Only entropy loss is used to encourage focused attention on important words.

        Returns:
            dict: Dictionary containing individual losses and total alignment loss
        """
        if not hasattr(self, 'Transformer') or not hasattr(self.Transformer, 'blocks'):
            device = next(self.parameters()).device
            return {'alignment_loss': torch.tensor(0.0, device=device)}

        # Collect attention maps from all transformer blocks
        all_attn_maps = []
        for block in self.Transformer.blocks:
            if hasattr(block, 'attn_maps') and block.attn_maps is not None:
                # attn_maps is tuple of (attn_map1, attn_map2) for two persons
                attn_map1, attn_map2 = block.attn_maps
                # Each is [B, nhead, n, M] where n=motion tokens, M=words
                # Average over heads to get [B, n, M]
                attn_map1 = attn_map1.mean(dim=1)  # [B, n, M]
                attn_map2 = attn_map2.mean(dim=1)  # [B, n, M]
                all_attn_maps.append(attn_map1)
                all_attn_maps.append(attn_map2)

        if len(all_attn_maps) == 0:
            device = next(self.parameters()).device
            return {'alignment_loss': torch.tensor(0.0, device=device)}

        # Stack all attention maps: [num_layers*2, B, n, M]
        attn_maps = torch.stack(all_attn_maps, dim=0)

        # Entropy Loss: Encourage focused attention on specific words
        # Goal: Each motion token should attend to specific words, not all uniformly
        # Compute entropy: -sum(p * log(p))
        attn_maps_mean = attn_maps.mean(dim=0)  # [B, n, M] - average across layers
        epsilon = 1e-8
        entropy = -(attn_maps_mean * torch.log(attn_maps_mean + epsilon)).sum(dim=-1)  # [B, n]
        entropy_loss = entropy.mean()

        # Total alignment loss = entropy only (coverage loss removed)
        alignment_loss = entropy_loss

        return {
            'alignment_loss': alignment_loss,
            'entropy_loss': entropy_loss
        }

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False, word_emb=None, word_mask=None):
        '''
        :param motion_ids: (b, seqlen)
        :padding_mask: (b, seqlen), all pad positions are TRUE else FALSE
        :param cond: (b, embed_dim) for text, (b, num_actions) for action
        :param force_mask: boolean
        :param word_emb: (b, M, d) word embeddings for cross-attention (optional)
        :param word_mask: (b, M) valid word positions (optional)
        :return:
            -logits: (b, num_token, seqlen)
        '''
        bs, n_tokens = motion_ids.shape

        cond = self.mask_cond(cond, force_mask=force_mask)

        motion_ids = torch.cat((motion_ids[:, :n_tokens//2],
                                self.sep_id * torch.ones((bs, 1), device=motion_ids.device, dtype=torch.long),
                                motion_ids[:, n_tokens//2:]), dim=-1)

        x = self.token_emb(motion_ids)

        # (b, seqlen, d) -> (seqlen, b, latent_dim)
        x = self.input_process(x)
        cond = self.cond_emb(cond) #(1, b, latent_dim)

        x = self.position_enc(x)

        padding_mask = torch.cat([padding_mask[:, :n_tokens//2],
                                torch.zeros_like(padding_mask[:, 0:1]),
                                padding_mask[:, n_tokens//2:]], dim=1) #(b, seqlen+1)

        # safety: 어떤 배치도 "전부 패딩"이 되지 않도록 보정
        all_pad = padding_mask.all(dim=1)
        if all_pad.any():
            padding_mask[all_pad, 0] = False

        # Pass word embeddings to transformer for word-motion cross-attention
        # CRITICAL FIX: MultiheadAttention uses batch_first=False, so permute word_emb
        # from [B, M, d] to [M, B, d] to match expected format
        if word_emb is not None:
            word_emb = word_emb.permute(1, 0, 2)  # [B, M, d] -> [M, B, d]

        output = self.Transformer(x, cond, src_key_padding_mask=padding_mask,
                                 word_emb=word_emb, word_mask=word_mask) #(seqlen, b, e)
        logits = self.output_process(output, cond) #(seqlen, b, e) -> (b, ntoken, seqlen)
        return logits

    def forward(self, ids, y, m_lens):
        '''
        :param ids: (b, n)
        :param y: raw text for cond_mode=text, (b, ) for cond_mode=action
        :m_lens: (b,)
        :return:
        '''

        bs, ntokens = ids.shape
        device = ids.device

        # ---- 배치 정렬: ids와 m_lens 길이가 반드시 같게 ----
        if isinstance(m_lens, (list, tuple)):
            m_lens = torch.as_tensor(m_lens, device=device)
        else:
            m_lens = m_lens.to(device)

        if m_lens.dim() != 1:
            m_lens = m_lens.view(-1)

        if m_lens.size(0) != bs:
            # 마지막 배치 등으로 인해 길이 어긋난 경우 강제 정렬
            min_bs = min(bs, m_lens.size(0))
            ids = ids[:min_bs]
            m_lens = m_lens[:min_bs]
            bs, ntokens = ids.shape  # 갱신

        # --- Correct non_pad_mask for [2 persons × nbp parts × length] ---
        B, T = ids.shape
        nbp = self.nbp                               # = 5
        L = max(T // (2 * nbp), 1)

        Ls = m_lens.clamp(min=1, max=L)              # [B]
        base_mask = lengths_to_mask(Ls, L)           # [B, 2*L] (True=non-pad)
        non_pad_mask = base_mask.repeat(1, nbp)      # [B, T]

        # 모든 토큰이 패딩(True) 되는 행 방지용 보정
        empty_rows = ~non_pad_mask.any(dim=1)
        if empty_rows.any():
            non_pad_mask[empty_rows, 0] = True

        # pad 위치는 pad_id로 채우기
        ids = torch.where(non_pad_mask, ids, torch.full_like(ids, self.pad_id))

        # non_pad_mask = lengths_to_mask(m_lens, max_len) #(b, n)
        # non_pad_mask = non_pad_mask.repeat(1, self.nbp)
        # # print(f">>> Pad mask: {non_pad_mask.reshape(2,2,-1).reshape(2,2,5,-1)}")

        # ids = torch.where(non_pad_mask, ids, self.pad_id)
        # # print(f">>> Padded ids: {ids.reshape(2,2,-1).reshape(2,2,5,-1)}")

        force_mask = False
        word_emb = None
        word_mask = None

        if self.cond_mode == 'text':
            # Get both sentence and word-level embeddings
            cond_vector, word_emb, word_mask = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")


        '''
        Prepare mask
        '''
        interaction_mask = torch.bernoulli(torch.tensor(self.opt.interaction_mask_prob)).bool().item()
        if interaction_mask:
            # 각 샘플별로 어떤 사람을 가릴지(앞/뒤 절반) 확률 0.5로 결정
            # DDP-aware: Single GPU와 동일한 난수 시퀀스 보장
            choose_first = ddp_aware_bernoulli(0.5, (bs,), device=device).bool()   # [bs]

            left_half  = torch.zeros((bs, ntokens // 2), device=device, dtype=torch.bool)  # 앞 사람
            right_half = torch.ones((bs,  ntokens // 2), device=device, dtype=torch.bool)  # 뒤 사람

            # sample-wise로 앞사람/뒷사람 마스킹 선택
            first_person_mask  = torch.where(choose_first.view(-1, 1), left_half,  right_half)  # [bs, ntokens//2]
            second_person_mask = torch.where(choose_first.view(-1, 1), right_half, left_half)   # [bs, ntokens//2]

            mask = torch.cat((first_person_mask, second_person_mask), dim=-1)  # [bs, ntokens]

            rand_mask_probs = torch.ones((bs,), device=device) * 0.5
            rand_time = self.noise_schedule_backward(rand_mask_probs)
        else:
            rand_time = uniform((bs,), device=device)
            rand_mask_probs = self.noise_schedule(rand_time)
            num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)
            # DDP-aware: Single GPU와 동일한 난수 시퀀스 보장
            batch_randperm = ddp_aware_rand((bs, ntokens), device=device).argsort(dim=-1)
            mask = batch_randperm < num_token_masked.unsqueeze(-1)

        # Positions to be MASKED must also be NON-PADDED
        #mask &= non_pad_mask
        # Positions to be MASKED must also be NON-PADDED
        if mask.size() != non_pad_mask.size():
            min_bs = min(mask.size(0), non_pad_mask.size(0))
            min_T  = min(mask.size(1), non_pad_mask.size(1))
            mask         = mask[:min_bs, :min_T]
            non_pad_mask = non_pad_mask[:min_bs, :min_T]
        mask = mask & non_pad_mask

        # Note this is our training target, not input
        # labels = torch.where(mask, ids, self.mask_id)

        # --- build labels ---
        # Make sure labels are valid: only non-pad positions keep ids, others set to mask_id
        labels = torch.where(non_pad_mask, ids, self.mask_id)

        # Also, if any pad_id slipped in, map them to mask_id so loss can ignore them
        labels = torch.where(labels == self.pad_id, torch.full_like(labels, self.mask_id), labels)

        x_ids = ids.clone()

        # Further Apply Bert Masking Scheme
        # Step 1: 10% replace with an incorrect token
        mask_rid = get_mask_subset_prob(mask, 0.1)
        # DDP-aware: Single GPU와 동일한 난수 시퀀스 보장
        bs, seq_len = x_ids.shape
        rand_id = ddp_aware_randint(0, self.opt.num_tokens, (bs, seq_len), device=x_ids.device, dtype=x_ids.dtype)
        x_ids = torch.where(mask_rid, rand_id, x_ids)
        # Step 2: 90% x 10% replace with correct token, and 90% x 88% replace with mask token
        mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)

        x_ids = torch.where(mask_mid, self.mask_id, x_ids)

        # Forward pass with word-motion cross-attention
        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask,
                                   word_emb=word_emb, word_mask=word_mask)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        # Compute alignment losses if word embeddings are provided
        # 251020 >> LOSS Delete Test
        # if word_emb is not None and self.training:
        #     #alignment_losses = self.compute_alignment_losses(word_mask)
        #     # Add weighted alignment loss to ce_loss (reduced from 0.1 to 0.01)
        #     ce_loss = ce_loss + 0.01 * alignment_losses['alignment_loss'] 

        logits = logits.permute(0,2,1) # B,ntokens,T -> B,T,ntokens

        if self.opt.step_unroll:
            su_ce_loss, su_pred_id, su_acc = self.step_unroll_forward(x_ids, mask_mid, labels, logits,
                                                                cond_vector, non_pad_mask, force_mask,
                                                                word_emb=word_emb, word_mask=word_mask)

            return ce_loss + (self.opt.step_unroll * su_ce_loss), (acc + self.opt.step_unroll*su_acc)/2, pred_id, su_pred_id, logits
        else:
            return ce_loss, acc, pred_id, None, logits

    def step_unroll_forward(self, prev_masked_ids, prev_mask, prev_labels, logits, cond_vector, non_pad_mask, force_mask, word_emb=None, word_mask=None):
        # print(f">>>>>>>>>>>> Step unroll >>>>>>>>>>>>>>>")
        total_timesteps = 20
        prev_rand_mask_probs = prev_mask.count_nonzero(dim = -1).float() / prev_mask.shape[-1]
        prev_rand_time = self.noise_schedule_backward(prev_rand_mask_probs)

        rand_time = (prev_rand_time + (1/(total_timesteps+1))).clamp(max=1)
        rand_mask_probs = self.noise_schedule(rand_time)

        probs = logits.softmax(dim=-1)
        scores, pred_ids = probs.max(dim=-1)
        scores = scores.masked_fill(~prev_mask, 1e5)

        sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
        ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
        num_token_masked = torch.round(rand_mask_probs * (scores.shape[-1])).clamp(min=1)

        mask = (ranks < num_token_masked.unsqueeze(-1))

        retained_preds = torch.logical_and(prev_mask == True,  mask == False)
        labels = torch.where(retained_preds, self.mask_id, prev_labels)

        x_ids = torch.where(retained_preds, pred_ids, prev_masked_ids)

        # Pass word embeddings through step_unroll forward pass as well
        step_unroll_logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask,
                                                word_emb=word_emb, word_mask=word_mask)
        
        # Debug check
        if torch.isnan(step_unroll_logits).any():
            print("NaN in logits!", step_unroll_logits)

        # mask_id (1024)가 라벨에 포함되는 것은 정상 - ignore_index로 처리됨
        # 디버깅을 위한 범위 체크 (mask_id 제외)
        valid_labels = labels[labels != self.mask_id]
        if valid_labels.numel() > 0 and valid_labels.max() >= step_unroll_logits.size(1):
            print(f"Warning: Valid label out of range! Max valid label: {valid_labels.max().item()}, Expected range: [0, {step_unroll_logits.size(1)-1}]")
            print(f"Labels shape: {labels.shape}, Logits shape: {step_unroll_logits.shape}")
            print(f"mask_id: {self.mask_id}, will be ignored in loss calculation")

        return cal_performance(step_unroll_logits, labels, ignore_index=self.mask_id)

    def forward_with_cond_scale(self,
                                motion_ids,
                                cond_vector,
                                padding_mask,
                                cond_scale=3,
                                force_mask=False,
                                word_emb=None,
                                word_mask=None):
        # bs = motion_ids.shape[0]
        # if cond_scale == 1:
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True,
                                     word_emb=word_emb, word_mask=word_mask)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                   word_emb=word_emb, word_mask=word_mask)
        if cond_scale == 1:
            return logits

        # Unconditional pass: force_mask=True removes text conditioning
        # But we still pass word_emb for consistency (it will be masked internally)
        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask, force_mask=True,
                                        word_emb=word_emb, word_mask=word_mask)

        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 timesteps: int,
                 cond_scale: int,
                 temperature=1,
                 topk_filter_thres=0.9,
                 gsample=False,
                 force_mask=False
                 ):

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)
        token_lengths = m_lens*2

        word_emb = None
        word_mask = None

        if self.cond_mode == 'text':
            # Get both sentence and word-level embeddings for better generation
            cond_vector, word_emb, word_mask = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        padding_mask = padding_mask.repeat(1, self.nbp)
        token_lengths = token_lengths*self.nbp

        # Start from all tokens being masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * token_lengths).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  word_emb=word_emb,
                                                  word_mask=word_mask)

            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            temperature = starting_temperature
            # temperature is annealed, gradually reducing temperature hence randomness
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    @torch.no_grad()
    @eval_decorator
    def generate_reaction(self,
                            conds,
                            motion1_ids,
                            m_lens,
                            timesteps: int,
                            cond_scale: int,
                            temperature=1,
                            topk_filter_thres=0.9,
                            gsample=False,
                            force_mask=False
                            ):

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)
        token_lengths = m_lens

        word_emb = None
        word_mask = None

        if self.cond_mode == 'text':
            with torch.no_grad():
                # Get both sentence and word-level embeddings for better generation
                cond_vector, word_emb, word_mask = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(batch_size, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        padding_mask = padding_mask.repeat(1, self.nbp)
        token_lengths = token_lengths*self.nbp

        # Start from all tokens being masked
        ids1 = torch.where(padding_mask[:, :padding_mask.shape[1]//2], self.pad_id, motion1_ids)
        ids = torch.where(padding_mask[:, :padding_mask.shape[1]//2], self.pad_id, self.mask_id)
        scores = torch.where(ids == self.mask_id, 0., 1e5)

        starting_temperature = temperature
        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps//2, device=device), reversed(range(timesteps))):
            # 0 < timestep < 1
            rand_mask_prob = self.noise_schedule(timestep)  # Tensor

            '''
            Maskout, and cope with variable length
            '''
            # fix: the ratio regarding lengths, instead of seq_len
            num_token_masked = torch.round(rand_mask_prob * token_lengths).clamp(min=1)  # (b, )

            # select num_token_masked tokens with lowest scores to be masked
            sorted_indices = scores.argsort(dim=1)  # (b, k), sorted_indices[i, j] = the index of j-th lowest element in scores on dim=1
            ranks = sorted_indices.argsort(dim=1)  # (b, k), rank[i, j] = the rank (0: lowest) of scores[i, j] on dim=1
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            '''
            Preparing input
            '''
            # (b, num_token, seqlen)
            logits = self.forward_with_cond_scale(torch.cat((ids1,ids), dim=-1), cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  word_emb=word_emb,
                                                  word_mask=word_mask)
            logits = logits[:, :, padding_mask.shape[1]//2:]
            
            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            # print(logits.shape, self.opt.num_tokens)
            # clean low prob token
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            '''
            Update ids
            '''
            # if force_mask:
            temperature = starting_temperature
            if gsample:  # use gumbel_softmax sampling
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)  # (b, seqlen)
            else:  # use multinomial sampling
                probs = F.softmax(filtered_logits, dim=-1)  # (b, seqlen, ntoken)
                pred_ids = Categorical(probs / temperature).sample()  # (b, seqlen)

            # print(pred_ids.max(), pred_ids.min())
            # if pred_ids.
            ids = torch.where(is_mask, pred_ids, ids)

            '''
            Updating scores
            '''
            probs_without_temperature = logits.softmax(dim=-1)  # (b, seqlen, ntoken)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1))  # (b, seqlen, 1)
            scores = scores.squeeze(-1)  # (b, seqlen)

            # We do not want to re-mask the previously kept tokens, or pad tokens
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.cat((ids1, ids), dim=-1)
        ids = torch.where(padding_mask, -1, ids)

        return ids

                        