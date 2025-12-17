# InterMask: 3D Human Interaction Generation via Collaborative Masked Modeling


![teaser_image](assets/Figure1.png)

---

## 1. Project Summary

This repository extends the InterMask baseline for **two-person interaction motion generation** with improvements aimed at:

- **Fine-grained text understanding** (e.g., “left” vs “right”, specific body parts)
- **Training efficiency** (multi-GPU training on common RTX-class hardware)
- **Experiment traceability** (optional Weights & Biases logging)

### 1.1 Baseline: InterMask Architecture

The baseline InterMask pipeline is a two-stage setup:

- **VQ-VAE**: tokenizes continuous motion into discrete motion tokens
- **Masked Transformer**: iteratively predicts motion tokens conditioned on text prompts

### 1.2 Key Contributions

#### 1.2.1 Word-Motion Cross Attention (Primary Contribution)
**Motivation**: sentence-level conditioning alone can miss word-level constraints (direction, body part, role-specific instructions).

**Idea**: insert a cross-attention block that lets **motion features attend to word-level text tokens** (e.g., from a CLIP text encoder), prior to the Inter-M Transformer.

#### 1.2.2 Multi-GPU Distributed Training (DDP)
Adds Distributed Data Parallel support for training VQ-VAE and Transformer with:
- consistent RNG/seed handling
- sequential batch partitioning (to preserve single-GPU-equivalent dynamics)

#### 1.2.3 WandB Integration
Optional logging hooks across training/evaluation for:
- losses, metrics, hyperparameters
- reproducible run tracking

---

## 2. Code instruction

> Goal: make the repository runnable end-to-end with **only the essential commands + core code snippets**.

### 2.1 Preparation

#### 2.1.1 Setup environment
```bash
conda env create -f environment.yml
conda activate hhi
```
Tested on Python 3.7.7 and PyTorch 1.13.1.

#### 2.1.2 Download checkpoints and evaluation models
```bash
# Download pre-trained VQ-VAE + Transformer checkpoints (InterHuman / Inter-X)
python prepare/download_models.py

# Download evaluation models (InterHuman)
bash prepare/download_evaluator.sh
```

If `gdown` fails:
```bash
rm -f ~/.cache/gdown/cookies.json
```

#### 2.1.3 Download SMPL-X (only for Inter-X)
Download SMPL-X models from the official website and place them under:
`./data/body_models/smplx/`

Example structure:
```text
data/body_models/smplx
├── SMPLX_FEMALE.npz
├── SMPLX_FEMALE.pkl
├── SMPLX_MALE.npz
├── SMPLX_MALE.pkl
├── SMPLX_NEUTRAL.npz
├── SMPLX_NEUTRAL.pkl
└── SMPLX_NEUTRAL_2020.npz
```

#### 2.1.4 Get datasets

**InterHuman**
- Follow the InterGen repository instructions.
- Place the dataset at `./data/InterHuman/` and unzip `motions_processed.zip`.

**Inter-X**
- Follow the Inter-X repository instructions.
- Place it at `./data/Inter-X_Dataset/` and unzip `processed/texts_processed.tar.gz`.

**Inter-X evaluation models**
After downloading Inter-X, move:
`./data/Inter-X_Dataset/text2motion/checkpoints/*` → `./checkpoints/hhi/`

---

### 2.2 Train your own models

> Important: Train **VQ-VAE first**, then train the **Inter-M Transformer**.

#### 2.2.1 Train VQ-VAE

**Single GPU**
```bash
# InterHuman
python train_vq.py --gpu_id 0 --dataset_name interhuman --name vq_test

# Inter-X
python train_vq.py --gpu_id 0 --dataset_name interx --batch_size 128 --name vq_test
```

**Multi-GPU (DDP)**
```bash
# Example: 4 GPUs
# !! Batch_size is per-GPU; effective = batch_size * num_gpus
# InterHuman
torchrun --nproc_per_node=4 train_vq.py --distributed --dataset_name interhuman --name vq_ddp_test

# Inter-X
torchrun --nproc_per_node=4 train_vq.py --distributed --dataset_name interx --batch_size 32 --name vq_ddp_test
```

#### 2.2.2 Train Inter-M Transformer

**Single GPU**
```bash
# InterHuman
python train_transformer.py --gpu_id 0 --dataset_name interhuman --name trans_test --vq_name vq_test

# Inter-X
python train_transformer.py --gpu_id 0 --dataset_name interx --batch_size 128 --name trans_test --vq_name vq_test
```

**Multi-GPU (DDP)**
```bash
torchrun --nproc_per_node=4 train_transformer.py --distributed --dataset_name interhuman --name trans_ddp_test --vq_name vq_ddp_test
```

---

### 2.3 Evaluation

```bash
python eval.py --gpu_id 0 --dataset_name interhuman --name trans_test --which_epoch best_fid
```

Common arguments:
- `--which_epoch`: `all | best_fid | best_top1 | latest | finest`
- `--time_steps`: transformer inference iterations (default: 20)
- `--cond_scales`: classifier-free guidance scale (default: 2)
- `--topkr`: ignore percentile of low-score tokens during inference (default: 0.9)

Logs are written to:
```text
./checkpoints/<dataset_name>/<name>/eval/evaluation_<which_epoch>_ts<time_steps>_cs<cond_scales>_topkr<topkr>.log
```

---

### 2.4 Core code snippets (minimal)

#### 2.4.1 Word-Motion Cross Attention (conceptual core)
Below is a minimal PyTorch-style block showing the core idea: **motion queries attending to word tokens**.

```python
import torch
import torch.nn as nn

class WordMotionCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, motion_feat, word_tokens, word_padding_mask=None):
        # motion_feat: (B, Tm, D)  motion features (e.g., concatenated two-person features)
        # word_tokens: (B, Tw, D)  word-level text embeddings
        # word_padding_mask: (B, Tw)  True for PAD positions
        q = self.norm_q(motion_feat)
        kv = self.norm_kv(word_tokens)
        out, _ = self.attn(q, kv, kv, key_padding_mask=word_padding_mask)
        return motion_feat + self.proj(out)
```

#### 2.4.2 DDP entry (what matters)
Use `torchrun` and make sure:
- batch size is **per GPU**
- random seeds / samplers are synchronized in distributed mode

```bash
torchrun --nproc_per_node=4 train_transformer.py --distributed ...
```

#### 2.4.3 WandB (what matters)
If enabled in your scripts, pass `--use_wandb` (and optionally `--wandb_name`).

```bash
python train_vq.py --gpu_id 0 --dataset_name interhuman --name vq_test --use_wandb --wandb_name my_vq_run
python train_transformer.py --gpu_id 0 --dataset_name interhuman --name trans_test --vq_name vq_test --use_wandb --wandb_name my_trans_run
python eval.py --gpu_id 0 --dataset_name interhuman --name trans_test --use_wandb --wandb_name my_eval_run
```

---

## 3. Demo

### 3.1 Run inference

```bash
python infer.py --gpu_id 0 --dataset_name interhuman --name trans_default
```

- Prompts are read from `./prompts.txt` (one prompt per line).
- Default output folder:
  `./checkpoints/<dataset_name>/<name>/animation_infer/`

Outputs:
- `keypoint_npy`: generated motions, shape `(nframe, 22, 9)` for each person
- `keypoint_mp4`: stick-figure animation (two viewpoints)

### 3.2 Example results (GIF)

<p align="center">
  <img src="assets/Demo.gif" width="600" />
</p>

*Sample two-person interaction motion generated from a text prompt.*

> Note: If you only have an MP4 (e.g., `assets/infer_best_fid_ts20_cs2_topkr0.9_02_00.mp4`), convert it to GIF once:
```bash
ffmpeg -i assets/infer_best_fid_ts20_cs2_topkr0.9_02_00.mp4 -vf "fps=15,scale=600:-1:flags=lanczos" -loop 0 assets/infer_best_fid_ts20_cs2_topkr0.9_02_00.gif
```

---

## 4. Conclusion and Future Work

### 4.1 Conclusion
We improve an InterMask-style two-person motion generation pipeline by strengthening **word-level text conditioning** (via Word-Motion Cross Attention), enabling **multi-GPU distributed training** for practical hardware, and supporting **experiment tracking** through optional WandB logging.

### 4.2 Future Work (Affordance scenarios)
A natural next step is to extend generation beyond text-only conditioning by integrating **affordance-aware constraints** from the environment. For example, in a **VR/AR interaction authoring** scenario, the model could use object affordances (graspable, pushable, sit-on-able) and scene geometry (surface normals, collision, reachable regions) to produce motions that are not only plausible but also *physically and functionally compatible* with the surrounding context.

Another scenario is **human-robot or human-digital-human collaboration**, where affordances can act as a shared interface between language and action. By grounding prompts like “hand over the cup” or “help them stand up” into affordance-driven intermediate targets (contact points, support regions, handover poses), the system could improve safety, interpretability, and controllability—especially for fine-grained interactions.

### 4.3 Acknowledgements
This repository builds on the InterMask / InterGen ecosystem and uses InterHuman and Inter-X datasets and their associated evaluation tooling. Please refer to the linked upstream repositories and dataset licenses for usage terms.
