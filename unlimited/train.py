"""
Train an ensemble of language models and evaluate running ensemble val loss.

Trains N models (default 20) with different random seeds, shuffling data each epoch.
After each model is trained, computes ensemble val loss by averaging logits across
all models trained so far. 
The reported ensemble metric excludes model 0, which is weaker (no distillation 
teacher, fewer epochs) and hurts ensemble quality.

Usage:
    torchrun --standalone --nproc_per_node=8 unlimited/train.py

Usage (two nodes):
    On each node, run:

    torchrun --nnodes=2 --nproc_per_node=8 --node_rank={0 or 1} \
        --master_addr=<node0_ip> --master_port=29500 \
        unlimited/train.py [OPTIONS]

    Training data and checkpoint paths must be on a
    shared filesystem visible to both nodes.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import math
import time
import json
import argparse
from types import SimpleNamespace
from functools import partial
from dataclasses import dataclass
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import wandb
import tiktoken

# =============================================================================
# CLI arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Train GPT ensemble")
parser.add_argument("--device-batch-size", type=int, default=2)
parser.add_argument("--num-epochs-model-0", type=int, default=16, help="Epochs for first model (defaults to --num-epochs)")
parser.add_argument("--num-epochs", type=int, default=32, help="Total epochs for models that are not model 0")
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--run", type=str, default=None)
parser.add_argument("--scalar-lr", type=float, default=0.1)
parser.add_argument("--matrix-lr", type=float, default=0.04)
parser.add_argument("--weight-decay", type=float, default=0.1)
parser.add_argument("--total-batch-size", type=int, default=524288)
parser.add_argument("--save-result", type=str, default="")
parser.add_argument("--n_layer", type=int, default=12)
parser.add_argument("--n_head", type=int, default=12)
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--lr_multiplier", type=float, default=0.25)
parser.add_argument("--input_bin", type=str, default=None)
parser.add_argument("--input_val_bin", type=str, default=None)
parser.add_argument("--output_json", type=str, default=None)
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--num-models", type=int, default=20, help="Number of ensemble members")
parser.add_argument("--checkpoint-base", type=str, default="checkpoints", help="Base directory for checkpoints")
parser.add_argument("--resume", type=str, default=None, help="Run ID to resume from (e.g. 20250226_143000)")
parser.add_argument("--distill-alpha", type=float, default=0.7, help="Weight for distillation loss (0=hard labels only, 1=soft labels only)")
parser.add_argument("--distill-temperature", type=float, default=1.0, help="Temperature for softening teacher logits")
parser.add_argument("--dupe-layers-start", type=int, default=15,
                    help="First decoder layer to duplicate (inclusive)")
parser.add_argument("--dupe-layers-end", type=int, default=25,
                    help="Last decoder layer to duplicate (exclusive)")
parser.add_argument("--dupe-fraction", type=float, default=1.0, help="Dupe layers activate for the last (1 - this) fraction of epochs. Default 1.0 disables dupe entirely.")
parser.add_argument("--ema-decays", type=str, default="0.95",
                    help="Comma-separated EMA decay rates, e.g. '0.999,0.9995,0.9998'")
parser.add_argument("--ema-start-frac", type=float, default=0.90,
                    help="Fraction of training after which to start EMA tracking")
parser.add_argument("--ensemble-mode", type=str, default="logit", choices=["prob", "logit"],
                    help="Ensemble averaging method: 'prob' averages softmax probabilities, 'logit' averages raw logits")
parser.add_argument("--optimizer", type=str, default="hybrid", choices=["hybrid", "muon", "adamw"],
                    help="'hybrid' = Muon for matrices + AdamW for rest (default), "
                         "'muon' = Muon for all trainable params, "
                         "'adamw' = pure AdamW for all params")
parser.add_argument("--completep", action="store_true", default=False,
                    help="Enable CompleteP: muP width scaling + 1/L depth scaling")
parser.add_argument("--mup-base-width", type=int, default=256,
                    help="Base width for muP LR scaling (only with --completep and --optimizer adamw)")
parser.add_argument("--ensemble-type", type=str, default="init_shuffle",
                    choices=["init", "init_shuffle"],
                    help="'init' = different model inits, same data order; "
                         "'init_shuffle' = different inits AND data orders (default)")
parser.add_argument("--no-compile", action="store_true", default=False,
                    help="Disable torch.compile (use eager mode; needed when Triton/Python.h unavailable)")
parser.add_argument("--no-distill", action="store_true", default=False,
                    help="Disable chain distillation; each model trains independently on hard labels only")
parser.add_argument("--data-fraction", type=float, default=1.0,
                    help="Use only this fraction of the training data (0.0, 1.0]. Smaller fraction + more epochs lets you study multi-epoch dynamics faster.")
parser.add_argument("--val-every-n-steps", type=int, default=0,
                    help="If >0, run val eval (individual + ensemble) every N outer training steps. "
                         "Default 0 = only at epoch boundaries. Expensive when small.")
args = parser.parse_args()

assert 0.0 < args.data_fraction <= 1.0, "--data-fraction must be in (0, 1]"

if args.no_compile:
    torch._dynamo.config.disable = True

# Workaround for torch 2.8 inductor dtype bug in pad_mm benchmark pass
import torch._inductor.config as _inductor_config
_inductor_config.shape_padding = False

if args.output_json and not args.save_result:
    args.save_result = args.output_json

# =============================================================================
# Hyperparameters
# =============================================================================

DEPTH = args.n_layer
N_EMBD = args.n_embd
N_HEAD = args.n_head
HEAD_DIM = N_EMBD // N_HEAD
MAX_SEQ_LEN = 2048
WINDOW_PATTERN = "SSSL"
TOTAL_BATCH_SIZE = args.total_batch_size
EVAL_TOKENS = 10_000_000
DATA_DIR = "fineweb_data"

BASE_MATRIX_LR = args.matrix_lr
BASE_SCALAR_LR = args.scalar_lr
BASE_EMBEDDING_LR = 0.15
BASE_UNEMBEDDING_LR = 0.002

_lr_mult = args.lr_multiplier if args.lr_multiplier is not None else 1.0
MATRIX_LR = BASE_MATRIX_LR * _lr_mult
UNEMBEDDING_LR = BASE_UNEMBEDDING_LR * _lr_mult
EMBEDDING_LR = BASE_EMBEDDING_LR * _lr_mult
SCALAR_LR = BASE_SCALAR_LR * _lr_mult

WEIGHT_DECAY = args.weight_decay
ADAM_BETAS = (0.8, 0.95)
WARMUP_RATIO = 0.0
WARMDOWN_RATIO = 0.2
FINAL_LR_FRAC = 0.0

# =============================================================================
# Utilities
# =============================================================================

def get_dist_info():
    if all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE")):
        return True, int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    return False, 0, 0, 1

def print0(s="", **kwargs):
    if int(os.environ.get('RANK', 0)) == 0:
        print(s, **kwargs)

class DummyWandb:
    def __init__(self): self.summary = {}
    def log(self, *a, **kw): pass
    def finish(self): pass

# =============================================================================
# EMA (Exponential Moving Average) for weight averaging
# =============================================================================

class EMATracker:
    """Maintains EMA shadow weights on CPU for memory efficiency."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {name: p.data.float().cpu().clone() for name, p in model.named_parameters()}
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model):
        self.num_updates += 1
        d = self.decay
        for name, p in model.named_parameters():
            self.shadow[name].lerp_(p.data.float().cpu(), 1 - d)

    def apply_to(self, model):
        """Copy EMA weights into model (for evaluation)."""
        for name, p in model.named_parameters():
            p.data.copy_(self.shadow[name].to(p.device, dtype=p.dtype))

    def state_dict(self):
        return dict(self.shadow)


def load_state_dict_into_model(model, state_dict):
    """Load a state dict into model, handling dtype conversion."""
    for name, p in model.named_parameters():
        if name in state_dict:
            p.data.copy_(state_dict[name].to(p.device, dtype=p.dtype))

# =============================================================================
# Flash Attention
# =============================================================================

def _load_fa3():
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        if major != 9:
            return None
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None

_fa3 = _load_fa3()

def _load_fa2():
    try:
        from flash_attn import flash_attn_func as _fa2_func
        return _fa2_func
    except (ImportError, OSError):
        return None

_fa2 = None if _fa3 is not None else _load_fa2()

# SDPA fallback: always available in torch >= 2.0
_use_sdpa = (_fa3 is None and _fa2 is None)


def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Flash Attention for training. Prefers FA3 > FA2 > PyTorch SDPA.
    q, k, v: (B, T, H, D) for FA3/FA2; transposed internally for SDPA.
    """
    if _fa3 is not None:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    if _fa2 is not None:
        return _fa2(q, k, v, causal=causal, window_size=window_size)
    # SDPA fallback: expects (B, H, T, D)
    q_s, k_s, v_s = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    # SDPA sliding window: only supported in torch >= 2.5 via enable_gqa; approximate with full attention if window requested
    out = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=causal)
    return out.transpose(1, 2)  # back to (B, T, H, D)

flash_attn = SimpleNamespace(flash_attn_func=flash_attn_func)

# =============================================================================
# GPT Model
# =============================================================================

@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 50257
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN
    dropout: float = 0.0
    completep: bool = False
    mup_base_width: int = 256
    optimizer: str = "hybrid"

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Attention gate: per-head gating to enable context-based no-op
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Attention gate: per-head sigmoid gate
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_channels])).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config, layer_idx, depth_scale=1.0):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        self.depth_scale = depth_scale

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.depth_scale * self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.depth_scale * self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        depth_scale = 1.0 / config.n_layer if config.completep else 1.0
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i, depth_scale=depth_scale) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.output_multiplier = config.mup_base_width / config.n_embd if config.completep else 1.0
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({str(i): nn.Linear(config.n_embd, kv_dim, bias=False) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # U-Net skip connections: encoder layer i → decoder layer (n_layer - 1 - i)
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._dupe_layers = None  # (start, end) or None

    def set_dupe_layers(self, start, end):
        assert start >= self.encoder_layers, "dupe layers must be decoder-only"
        assert end <= self.config.n_layer
        self._dupe_layers = (start, end)
        print0(f"Dupe layers {start}-{end-1}: decoder layers repeated with skip connections")

    @torch.no_grad()
    def init_weights(self, convert_embed=True):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
        self.skip_weights.fill_(1.0)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        if convert_embed and self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
        self._dupe_layers = None

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def setup_optimizer(self):
        ddp, rank, local_rank, world_size = get_dist_info()
        matrix_params = list(self.transformer.h.parameters()) + list(self.ve_projs.parameters())
        ve_params = []
        embed_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        skip_params = [self.skip_weights]

        if self.config.optimizer == 'adamw':
            # Pure AdamW: AdamW for everything including matrix params
            # muP width scaling: scale matrix LR inversely with width
            matrix_lr = MATRIX_LR * (self.config.mup_base_width / self.config.n_embd) if self.config.completep else MATRIX_LR
            param_groups = [
                dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
                dict(kind='adamw', params=embed_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
                dict(kind='adamw', params=ve_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
                dict(kind='adamw', params=resid_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=skip_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=matrix_params, lr=matrix_lr, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
            ]
        elif self.config.optimizer == 'muon':
            # Pure Muon: Muon for everything (matrix params, embeddings, LM head)
            # Scalars still use AdamW (Muon requires 2D+ tensors for orthogonalization)
            all_muon_params = matrix_params + list(self.transformer.wte.parameters()) + list(self.lm_head.parameters())
            param_groups = [
                dict(kind='adamw', params=ve_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
                dict(kind='adamw', params=resid_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=skip_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            ]
            for shape in sorted({p.shape for p in all_muon_params}):
                group_params = [p for p in all_muon_params if p.shape == shape]
                param_groups.append(dict(kind='muon', params=group_params, lr=MATRIX_LR,
                                         momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY))
        else:
            # Hybrid (default): Muon for matrix params, AdamW for embeddings/scalars/LM-head
            param_groups = [
                dict(kind='adamw', params=lm_head_params, lr=UNEMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
                dict(kind='adamw', params=embed_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
                dict(kind='adamw', params=ve_params, lr=EMBEDDING_LR, betas=ADAM_BETAS, eps=1e-10, weight_decay=WEIGHT_DECAY),
                dict(kind='adamw', params=resid_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=x0_params, lr=SCALAR_LR, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
                dict(kind='adamw', params=skip_params, lr=SCALAR_LR * 0.01, betas=ADAM_BETAS, eps=1e-10, weight_decay=0.0),
            ]
            for shape in sorted({p.shape for p in matrix_params}):
                group_params = [p for p in matrix_params if p.shape == shape]
                param_groups.append(dict(kind='muon', params=group_params, lr=MATRIX_LR,
                                         momentum=0.95, ns_steps=5, beta2=0.95, weight_decay=WEIGHT_DECAY))

        optimizer = DistMuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _run_decoder_layers(self, x, x0, cos_sin, encoder_outputs, start, end):
        """Run decoder layers [start, end), with U-Net skip connections."""
        for i in range(start, end):
            # Encoder layer j connects to decoder layer (n_layer - 1 - j)
            j = self.config.n_layer - 1 - i
            if 0 <= j < self.encoder_layers:
                x = x + self.skip_weights[i - self.encoder_layers] * encoder_outputs[j]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
        return x

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x

        # Encoder half: run layers and collect outputs for skip connections
        encoder_outputs = []
        for i in range(self.encoder_layers):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = self.transformer.h[i](x, ve, cos_sin, self.window_sizes[i])
            encoder_outputs.append(x)

        # Decoder half
        dupe = self._dupe_layers
        if dupe is None:
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        self.encoder_layers, self.config.n_layer)
        else:
            # First pass: encoder boundary through end of dupe range
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        self.encoder_layers, dupe[1])
            # Replay 1
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        dupe[0], dupe[1])
            # Replay 2
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        dupe[0], dupe[1])
            # Replay 3
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        dupe[0], dupe[1])
            # Replay 4
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        dupe[0], dupe[1])
            # Remaining decoder layers
            x = self._run_decoder_layers(x, x0, cos_sin, encoder_outputs,
                                        dupe[1], self.config.n_layer)

        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = logits * self.output_multiplier
        logits = 15 * torch.tanh(logits / 15)
        if targets is not None:
            if loss_reduction == 'none':
                return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction='none')
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return logits

    def forward_logits(self, idx):
        """Forward pass returning only logits (no loss computation)."""
        return self.forward(idx, targets=None)

# =============================================================================
# Optimizer: MuonAdamW
# =============================================================================

polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    dtype = p.dtype
    p.mul_(1 - lr_t * wd_t)
    exp_avg.lerp_(grad, (1 - beta1_t).to(dtype))
    exp_avg_sq.lerp_(grad.square(), (1 - beta2_t).to(dtype))
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    p.add_(exp_avg / ((exp_avg_sq / bias2).sqrt() + eps_t), alpha=-(lr_t / bias1))

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, (1 - momentum).to(momentum_buffer.dtype))
    g = stacked_grads.lerp_(momentum_buffer, momentum.to(stacked_grads.dtype))
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), (1 - beta2).to(second_momentum_buffer.dtype))
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


class DistMuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        self._adamw_step_t = torch.tensor(0.0)
        self._adamw_lr_t = torch.tensor(0.0)
        self._adamw_beta1_t = torch.tensor(0.0)
        self._adamw_beta2_t = torch.tensor(0.0)
        self._adamw_eps_t = torch.tensor(0.0)
        self._adamw_wd_t = torch.tensor(0.0)
        self._muon_momentum_t = torch.tensor(0.0)
        self._muon_lr_t = torch.tensor(0.0)
        self._muon_wd_t = torch.tensor(0.0)
        self._muon_beta2_t = torch.tensor(0.0)

    def _reduce_adamw(self, group, world_size):
        infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                assert grad.shape[0] % world_size == 0
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=infos)

    def _reduce_muon(self, group, world_size):
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        stacked_grads = torch.empty(padded, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(torch.stack([p.grad for p in params]))
        if len(params) < padded:
            stacked_grads[len(params):].zero_()
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()
        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group, info, gather_list, rank, world_size):
        for p in group['params']:
            pinfo = info['param_infos'][p]
            pinfo['future'].wait()
            state = self.state[p]
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(p_slice, pinfo['grad_slice'], state['exp_avg'], state['exp_avg_sq'],
                           self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                           self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group, info, gather_list, rank):
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            s = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(s, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        updated = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        if num_owned > 0:
            owned = torch.stack([params[start_idx + i] for i in range(num_owned)])
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(info['grad_chunk'][:num_owned], owned,
                          state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                          self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                          group["ns_steps"], red_dim)
            updated[:num_owned].copy_(owned)
        if num_owned < chunk_size:
            updated[num_owned:].zero_()
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    @torch.no_grad()
    def step(self):
        rank, world_size = dist.get_rank(), dist.get_world_size()
        reduce_infos = []
        for group in self.param_groups:
            if group['kind'] == 'adamw': reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon': reduce_infos.append(self._reduce_muon(group, world_size))
        gather_list = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw': self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon': self._compute_muon(group, info, gather_list, rank)
        for info in gather_list:
            info["future"].wait()
            if info.get("params") is not None:
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

# =============================================================================
# Dataloader with epoch shuffling
# =============================================================================

class DataLoader:
    """Pre-tokenized chunk dataloader with per-epoch shuffling."""

    def __init__(self, filepath, B, T, device="cuda", seed=42, quiet=False, reshuffle_per_epoch=True, data_fraction=1.0):
        data = torch.load(filepath, weights_only=True)
        chunks = data['chunks']
        valid_counts = data['valid_counts']
        file_B = data['batch_size']
        sequence_size = data['sequence_size']
        assert sequence_size == T + 1, f"Data sequence_size {sequence_size} != T+1={T+1}"

        all_seqs = []
        for chunk, vc in zip(chunks, valid_counts):
            rows = chunk.view(file_B, sequence_size)[:vc]
            all_seqs.append(rows)
        all_seqs = torch.cat(all_seqs, dim=0).long()

        # Truncate to `data_fraction` of available data (deterministic: uses first N sequences).
        # A fixed slice is fine since sequences in the .pt file were already shuffled at prep time.
        if data_fraction < 1.0:
            keep = int(len(all_seqs) * data_fraction)
            all_seqs = all_seqs[:keep]

        _, rank, _, world_size = get_dist_info()
        seqs_per_step = B * world_size
        num_steps = len(all_seqs) // seqs_per_step
        usable = num_steps * seqs_per_step

        self.all_seqs = all_seqs[:usable]  # (usable, T+1) — keep flat for reshuffling
        self.B = B
        self.world_size = world_size
        self.rank = rank
        self.num_steps = num_steps
        self.seqs_per_step = seqs_per_step
        self.total_tokens = usable * T
        self.device = device
        self.seed = seed
        self.quiet = quiet
        self.reshuffle_per_epoch = reshuffle_per_epoch
        self.pos = 0
        self.epoch = 1
        self._shuffle_and_shard()

    def _shuffle_and_shard(self):
        """Shuffle all sequences and shard for this rank."""
        g = torch.Generator()
        # Fixed shuffle if reshuffle_per_epoch=False; advances per epoch otherwise.
        # Use multiplicative seeding so (seed_a, epoch_b) is independent from (seed_b, epoch_a).
        shuffle_seed = self.seed * 10_000 + (self.epoch if self.reshuffle_per_epoch else 0)
        g.manual_seed(shuffle_seed)
        perm = torch.randperm(len(self.all_seqs), generator=g)
        shuffled = self.all_seqs[perm]
        # Reshape: (num_steps, world_size, B, T+1)
        shaped = shuffled.view(self.num_steps, self.world_size, self.B, -1)
        self.rank_data = shaped[:, self.rank].contiguous()

    def __iter__(self):
        return self

    def __next__(self):
        if self.pos >= self.num_steps:
            self.pos = 0
            self.epoch += 1
            if not self.quiet:
                print0(f"Starting epoch {self.epoch}")
            self._shuffle_and_shard()  # reshuffle for new epoch
        batch = self.rank_data[self.pos].to(self.device, non_blocking=True)
        self.pos += 1
        return batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), self.epoch

# =============================================================================
# Evaluation helpers
# =============================================================================

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """Compute bits per byte and mean cross-entropy loss."""
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    total_loss = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_tokens = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none').view(-1)
        y = y.view(-1)
        mask = y != -1
        total_loss += loss2d[mask].sum()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[y]
        total_nats += (loss2d * (num_bytes2d > 0)).sum()
        total_bytes += num_bytes2d.sum()
    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


@torch.no_grad()
def evaluate_ensemble_bpb(checkpoint_paths, config, token_bytes, device, autocast_ctx,
                          mode="prob"):
    """
    Compute ensemble val loss by averaging across all checkpoints.

    mode="prob":  mean(softmax(logits_i))       -> loss = -log(avg_prob[target])
    mode="logit": softmax(mean(logits_i))       -> loss = cross_entropy(avg_logits, target)
    """
    num_models = len(checkpoint_paths)
    print0(f"  Loading {num_models} model(s) into GPU memory... (ensemble mode: {mode})")

    # Load all models onto GPU
    ensemble_models = []
    for ckpt_path in checkpoint_paths:
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device=device)
        model.init_weights(convert_embed=False)  # initializes rotary buffers only; skip bfloat16 embed cast
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
        model.eval()
        ensemble_models.append(model)
        del state_dict

    # Use B=1 for ensemble eval to save memory (N models loaded simultaneously)
    B_ensemble = 1
    val_loader = DataLoader(
        args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt"),
        B_ensemble, MAX_SEQ_LEN, device=device, seed=0, quiet=True,
    )
    _, _, _, ddp_world_size = get_dist_info()
    ensemble_eval_steps = EVAL_TOKENS // (B_ensemble * MAX_SEQ_LEN * ddp_world_size)

    total_nats = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(val_loader)
    for _ in range(ensemble_eval_steps):
        x, y, _ = next(batch_iter)
        flat_y = y.view(-1)
        mask = flat_y != -1

        if mode == "logit":
            # Average raw logits, then compute loss
            logit_sum = None
            for model in ensemble_models:
                with autocast_ctx:
                    logits = model.forward_logits(x).float()
                flat_logits = logits.view(-1, logits.size(-1))
                if logit_sum is None:
                    logit_sum = torch.zeros_like(flat_logits)
                logit_sum += flat_logits
                del logits, flat_logits
            avg_logits = logit_sum / num_models
            loss_per_pos = F.cross_entropy(avg_logits, flat_y.clamp(min=0), reduction='none')
            del logit_sum, avg_logits
        else:
            # Average probabilities, then compute loss
            target_prob_sum = torch.zeros(flat_y.size(0), dtype=torch.float64, device=device)
            for model in ensemble_models:
                with autocast_ctx:
                    logits = model.forward_logits(x).float()
                probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
                target_prob_sum += probs.gather(1, flat_y.clamp(min=0).unsqueeze(-1)).squeeze(-1).double()
                del logits, probs
            avg_prob = target_prob_sum / num_models
            loss_per_pos = -torch.log(avg_prob + 1e-10)
            del target_prob_sum, avg_prob

        total_loss += loss_per_pos[mask].sum().double()
        total_tokens += mask.sum()

        num_bytes2d = token_bytes[flat_y.clamp(min=0)]
        total_nats += (loss_per_pos[mask].double() * (num_bytes2d[mask] > 0).double()).sum()
        total_bytes += num_bytes2d[mask].sum()

    # Cleanup all models
    del ensemble_models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


@torch.no_grad()
def evaluate_ensemble_in_memory(ensemble_models, val_loader, eval_steps, token_bytes,
                                 device, autocast_ctx, mode="prob"):
    """Same as evaluate_ensemble_bpb but models are already loaded in memory.

    Args:
        ensemble_models: list of already-built GPT models (in eval mode)
        val_loader: iterable yielding (x, y, epoch) batches
        eval_steps: number of batches to evaluate
    """
    num_models = len(ensemble_models)
    for m in ensemble_models:
        m.eval()

    total_nats = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_bytes = torch.tensor(0, dtype=torch.int64, device=device)
    total_loss = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_tokens = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(val_loader)
    for _ in range(eval_steps):
        x, y, _ = next(batch_iter)
        flat_y = y.view(-1)
        mask = flat_y != -1

        if mode == "logit":
            logit_sum = None
            for m in ensemble_models:
                with autocast_ctx:
                    logits = m.forward_logits(x).float()
                flat_logits = logits.view(-1, logits.size(-1))
                if logit_sum is None:
                    logit_sum = torch.zeros_like(flat_logits)
                logit_sum += flat_logits
                del logits, flat_logits
            avg_logits = logit_sum / num_models
            loss_per_pos = F.cross_entropy(avg_logits, flat_y.clamp(min=0), reduction='none')
            del logit_sum, avg_logits
        else:
            target_prob_sum = torch.zeros(flat_y.size(0), dtype=torch.float64, device=device)
            for m in ensemble_models:
                with autocast_ctx:
                    logits = m.forward_logits(x).float()
                probs = F.softmax(logits.view(-1, logits.size(-1)), dim=-1)
                target_prob_sum += probs.gather(1, flat_y.clamp(min=0).unsqueeze(-1)).squeeze(-1).double()
                del logits, probs
            avg_prob = target_prob_sum / num_models
            loss_per_pos = -torch.log(avg_prob + 1e-10)
            del target_prob_sum, avg_prob

        total_loss += loss_per_pos[mask].sum().double()
        total_tokens += mask.sum()
        num_bytes2d = token_bytes[flat_y.clamp(min=0)]
        total_nats += (loss_per_pos[mask].double() * (num_bytes2d[mask] > 0).double()).sum()
        total_bytes += num_bytes2d[mask].sum()

    if dist.is_initialized():
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    total_nats, total_bytes = total_nats.item(), total_bytes.item()
    total_loss, total_tokens = total_loss.item(), total_tokens.item()
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float('inf')
    loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return bpb, loss


# =============================================================================
# Teacher loading for distillation
# =============================================================================

def load_teacher_models(checkpoint_paths, config, device):
    """Load previously trained models as frozen teachers for distillation."""
    teachers = []
    for ckpt_path in checkpoint_paths:
        with torch.device("meta"):
            m = GPT(config)
        m.to_empty(device=device)
        m.init_weights(convert_embed=False)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        m.load_state_dict(sd)
        m.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
        m.eval()
        teachers.append(m)
        del sd
    return teachers


# =============================================================================
# Training one model
# =============================================================================

@torch.no_grad()
def evaluate_distill_val(student, teacher, batches, steps, autocast_ctx, alpha, temperature, device):
    """Compute val KL loss, combined loss, and teacher CE loss."""
    total_student_ce = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_kl        = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_teacher_ce = torch.tensor(0.0, dtype=torch.float64, device=device)
    total_tokens    = torch.tensor(0, dtype=torch.int64, device=device)

    batch_iter = iter(batches)
    for _ in range(steps):
        x, y, _ = next(batch_iter)
        with autocast_ctx:
            student_logits = student.forward_logits(x).float()
            teacher_logits = teacher.forward_logits(x).float()

        flat_s = student_logits.view(-1, student_logits.size(-1))
        flat_t = teacher_logits.view(-1, teacher_logits.size(-1))
        flat_y = y.view(-1)
        mask = flat_y != -1

        student_ce_sum  = F.cross_entropy(flat_s, flat_y, ignore_index=-1, reduction='sum')
        teacher_ce_sum  = F.cross_entropy(flat_t, flat_y, ignore_index=-1, reduction='sum')
        T = temperature
        kl_sum = F.kl_div(
            F.log_softmax(flat_s[mask] / T, dim=-1),
            F.softmax(flat_t[mask] / T, dim=-1),
            reduction='sum',
        ) * (T * T)

        total_student_ce  += student_ce_sum.double()
        total_kl          += kl_sum.double()
        total_teacher_ce  += teacher_ce_sum.double()
        total_tokens      += mask.sum()

    if dist.is_initialized():
        for t in [total_student_ce, total_kl, total_teacher_ce, total_tokens]:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    n = total_tokens.item()
    if n == 0:
        return float('inf'), float('inf'), float('inf')

    val_kl         = total_kl.item() / n
    val_teacher_ce = total_teacher_ce.item() / n
    val_combined   = (1 - alpha) * (total_student_ce.item() / n) + alpha * val_kl
    return val_kl, val_combined, val_teacher_ce


def train_single_model(model_idx, seed, device, config, autocast_ctx, token_bytes,
                       wandb_run, ddp, ddp_world_size, checkpoint_dir,
                       teacher_checkpoint_paths=None, num_epochs=None):
    """Train a single model with the given seed. Returns path to saved checkpoint.

    If teacher_checkpoint_paths is non-empty, trains with knowledge distillation:
    each model learns from both the hard labels and the soft logits of the
    immediately preceding model (chain distillation). Only one teacher is loaded
    at a time, keeping memory usage constant regardless of ensemble size.

    After training, EMA-blended weights are evaluated and the best weights
    (final or blended) are saved to the checkpoint.
    """
    print0(f"\n{'='*60}")
    print0(f"Training model {model_idx + 1} with seed {seed}")
    print0(f"{'='*60}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    # Build model
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()

    # Keep reference to uncompiled model for dupe layer activation and EMA
    orig_model = model

    # Compile
    compiled_model = model if args.no_compile else torch.compile(model, dynamic=False)


    # Optimizer
    optimizer = compiled_model.setup_optimizer()

    # Dataloaders
    _train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
    # init_shuffle: each model sees different data order (default, current behavior)
    # init: all models see same data order, only model inits differ
    data_seed = seed if args.ensemble_type == "init_shuffle" else 42
    train_loader = DataLoader(_train_path, args.device_batch_size, MAX_SEQ_LEN, device=device, seed=data_seed)
    x, y, current_epoch = next(train_loader)

    # Training config
    normal_device_batch_size = args.device_batch_size
    dupe_device_batch_size = args.device_batch_size // 2  # used during dupe for models 1+

    tokens_per_fwdbwd = normal_device_batch_size * MAX_SEQ_LEN * ddp_world_size
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

    dupe_tokens_per_fwdbwd = dupe_device_batch_size * MAX_SEQ_LEN * ddp_world_size
    assert TOTAL_BATCH_SIZE % dupe_tokens_per_fwdbwd == 0
    dupe_grad_accum_steps = TOTAL_BATCH_SIZE // dupe_tokens_per_fwdbwd

    print0(f"  [model {model_idx+1}] Grad accum steps: {grad_accum_steps} (normal), {dupe_grad_accum_steps} (dupe, models 1+)")
    TOKENS_PER_EPOCH = train_loader.total_tokens
    num_iterations = round(TOKENS_PER_EPOCH * num_epochs / TOTAL_BATCH_SIZE)

    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    # Dupe layers: activate for the last 25% of epochs
    dupe_start_epoch = math.ceil(args.dupe_fraction * num_epochs) + 1  # epoch number (1-indexed) to activate
    dupe_active = False
    print0(f"  [model {model_idx+1}] Dupe layers will activate at epoch {dupe_start_epoch} (of {num_epochs})")

    # EMA setup
    ema_decays = [float(d) for d in args.ema_decays.split(",") if d.strip()] if args.ema_decays else []
    ema_start_step = round(args.ema_start_frac * num_iterations)
    ema_trackers = []
    ema_initialized = False
    if ema_decays:
        print0(f"  [model {model_idx+1}] EMA decays: {ema_decays}, starting at step {ema_start_step} ({args.ema_start_frac*100:.0f}% of training)")

    # LR schedule
    def get_lr_multiplier(it):
        warmup = round(WARMUP_RATIO * num_iterations)
        warmdown = round(WARMDOWN_RATIO * num_iterations)
        if it < warmup: return (it + 1) / warmup
        elif it <= num_iterations - warmdown: return 1.0
        else:
            progress = (num_iterations - it) / warmdown
            return progress + (1 - progress) * FINAL_LR_FRAC

    def get_muon_momentum(it):
        return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95

    # Training loop
    step = 0
    epoch_step = 0  # step within current epoch (resets each epoch)
    min_val_bpb = float("inf")
    min_val_loss = float("inf")
    epochs_without_improvement = 0
    smooth_train_loss = 0
    smooth_train_hard_loss = 0  # EMA for hard CE component
    smooth_train_kl_loss = 0    # EMA for KL distillation component
    epoch_loss_sum = 0.0   # accumulator for per-epoch mean train loss
    epoch_loss_count = 0
    total_training_time = 0
    eval_steps = EVAL_TOKENS // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size)

    # Load teacher models for distillation (empty list = no distillation)
    teacher_models = []
    if teacher_checkpoint_paths:
        print0(f"  [model {model_idx+1}] Loading {len(teacher_checkpoint_paths)} teacher model(s) for distillation...")
        teacher_models = load_teacher_models(teacher_checkpoint_paths, config, device)
        print0(f"  [model {model_idx+1}] Teachers loaded.")

    # Enable GC for fresh model
    gc.enable()
    gc.collect()

    compiled_model.train()
    while current_epoch <= num_epochs:
        # Activate dupe layers for the last 25% of training
        if not dupe_active and current_epoch >= dupe_start_epoch:
            print0(f"\n  [model {model_idx+1}] === Enabling dupe-layers at epoch {current_epoch} ===")
            orig_model.set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
            compiled_model = orig_model if args.no_compile else torch.compile(orig_model, dynamic=False)
            dupe_active = True
            gc.enable(); gc.collect()

            if model_idx >= 1:
                print0(f"  [model {model_idx+1}] Switching to dupe batch size {dupe_device_batch_size} "
                    f"(grad_accum_steps: {grad_accum_steps} -> {dupe_grad_accum_steps})")
                train_loader = DataLoader(_train_path, dupe_device_batch_size, MAX_SEQ_LEN, device=device, seed=data_seed)
                train_loader.epoch = current_epoch
                train_loader._shuffle_and_shard()
                x, y, current_epoch = next(train_loader)
                grad_accum_steps = dupe_grad_accum_steps

        synchronize()
        t0 = time.time()
        train_hard_loss = None
        train_kl_loss = None
        for micro_step in range(grad_accum_steps):
            if teacher_models:
                # --- Chain distillation loss ---
                with torch.inference_mode():
                    with autocast_ctx:
                        teacher_logits = teacher_models[0].forward_logits(x).float()

                with autocast_ctx:
                    student_logits = compiled_model(x)

                flat_s = student_logits.view(-1, student_logits.size(-1))
                flat_t = teacher_logits.view(-1, teacher_logits.size(-1))
                flat_y = y.view(-1)
                mask = flat_y != -1

                hard_loss = F.cross_entropy(flat_s, flat_y, ignore_index=-1)

                T = args.distill_temperature
                kl_loss = F.kl_div(
                    F.log_softmax(flat_s[mask] / T, dim=-1),
                    F.softmax(flat_t[mask] / T, dim=-1),
                    reduction='batchmean',
                ) * (T * T)

                loss = (1 - args.distill_alpha) * hard_loss + args.distill_alpha * kl_loss
                train_hard_loss = hard_loss.detach()
                train_kl_loss = kl_loss.detach()
                del teacher_logits
            else:
                # --- Standard cross-entropy loss ---
                with autocast_ctx:
                    loss = compiled_model(x, y)

            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()
            x, y, epoch = next(train_loader)

        lrm = get_lr_multiplier(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group['kind'] == 'muon':
                group["momentum"] = get_muon_momentum(step)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g['params']], max_norm=1.0)
        optimizer.step()
        compiled_model.zero_grad(set_to_none=True)
        train_loss_f = train_loss.item()
        synchronize()
        dt = time.time() - t0
        toks_per_sec = TOTAL_BATCH_SIZE / dt

        step += 1
        epoch_step += 1
        epoch_loss_sum += train_loss_f
        epoch_loss_count += 1

        # EMA update (every 10 steps to minimize CPU copy overhead)
        if ema_decays and step >= ema_start_step and step % 10 == 0:
            if not ema_initialized:
                print0(f"  [model {model_idx+1}] Initializing {len(ema_decays)} EMA tracker(s) at step {step}")
                ema_trackers = [EMATracker(orig_model, d) for d in ema_decays]
                ema_initialized = True
            for ema in ema_trackers:
                ema.update(orig_model)

        # Logging
        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased = smooth_train_loss / (1 - ema_beta**epoch_step)
        pct = 100 * step / num_iterations
        if step > 10:
            total_training_time += dt
        dupe_str = " [DUPE]" if dupe_active else ""
        if step % 50 == 0 or step == 1:
            print0(f"  [model {model_idx+1}] step {step:05d} ({pct:.2f}%) | loss: {debiased:.6f} | {toks_per_sec:.0f} tok/s{dupe_str}")

        log_dict = {
            "step": step,
            f"model_{model_idx+1}/train_loss": debiased,
            f"model_{model_idx+1}/train_loss_raw": train_loss_f,
            f"model_{model_idx+1}/epoch": current_epoch,
            f"model_{model_idx+1}/epoch_step": epoch_step,
            f"model_{model_idx+1}/tokens_seen": (current_epoch - 1) * TOKENS_PER_EPOCH + epoch_step * TOTAL_BATCH_SIZE,
            "model_idx": model_idx,
            "tokens_per_sec": toks_per_sec,
        }

        # Log decomposed distillation train losses when teacher is present
        if train_hard_loss is not None:
            smooth_train_hard_loss = ema_beta * smooth_train_hard_loss + (1 - ema_beta) * train_hard_loss.item()
            smooth_train_kl_loss   = ema_beta * smooth_train_kl_loss   + (1 - ema_beta) * train_kl_loss.item()
            debiased_hard = smooth_train_hard_loss / (1 - ema_beta**step)
            debiased_kl   = smooth_train_kl_loss   / (1 - ema_beta**step)
            log_dict[f"model_{model_idx+1}/train_hard_loss"] = debiased_hard
            log_dict[f"model_{model_idx+1}/train_kl_loss"]   = debiased_kl

        wandb_run.log(log_dict)

        # Epoch sync
        if ddp:
            epoch_tensor = torch.tensor([epoch], dtype=torch.long, device=device)
            dist.all_reduce(epoch_tensor, op=dist.ReduceOp.MAX)
            epoch = epoch_tensor.item()

        # Epoch boundary: evaluate
        if epoch != current_epoch:
            # Compute per-epoch mean train loss before resetting
            epoch_mean_train_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0
            print0(f"  [model {model_idx+1}] Epoch {current_epoch} | Mean Train Loss: {epoch_mean_train_loss:.6f} ({epoch_loss_count} steps)")

            compiled_model.eval()
            _val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")

            # Standard CE val metrics
            val_loader = DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN, device=device, seed=0, quiet=True)
            with autocast_ctx:
                val_bpb, val_loss = evaluate_bpb(compiled_model, val_loader, eval_steps, token_bytes)
            print0(f"  [model {model_idx+1}] Epoch {current_epoch} | Val BPB: {val_bpb:.6f} | Val Loss: {val_loss:.6f}")

            log_dict = {
                "step": step,
                f"model_{model_idx+1}/epoch": current_epoch,
                f"model_{model_idx+1}/tokens_seen": (current_epoch) * TOKENS_PER_EPOCH,
                f"model_{model_idx+1}/epoch_mean_train_loss": epoch_mean_train_loss,
                f"model_{model_idx+1}/val_bpb": val_bpb,
                f"model_{model_idx+1}/val_loss": val_loss,
            }

            # Distillation val metrics (only when a teacher is present)
            if teacher_models:
                val_loader2 = DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN, device=device, seed=0, quiet=True)
                val_kl, val_combined, teacher_val_ce = evaluate_distill_val(
                    student=compiled_model,
                    teacher=teacher_models[0],
                    batches=val_loader2,
                    steps=eval_steps,
                    autocast_ctx=autocast_ctx,
                    alpha=args.distill_alpha,
                    temperature=args.distill_temperature,
                    device=device,
                )
                print0(f"  [model {model_idx+1}] Epoch {current_epoch} | Val KL: {val_kl:.6f} | Val Combined: {val_combined:.6f} | Teacher Val CE: {teacher_val_ce:.6f}")
                log_dict.update({
                    f"model_{model_idx+1}/val_kl": val_kl,
                    f"model_{model_idx+1}/val_combined": val_combined,
                    f"model_{model_idx+1}/teacher_val_ce": teacher_val_ce,
                })

            wandb_run.log(log_dict)

            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
                min_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if args.patience >= 0 and epochs_without_improvement >= args.patience:
                    print0(f"  [model {model_idx+1}] Early stopping")
                    break
            compiled_model.train()
            current_epoch = epoch
            # Reset per-epoch trackers
            epoch_step = 0
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            smooth_train_loss = 0  # reset EMA so it doesn't bleed across epochs

        if step == 1:
            gc.collect(); gc.freeze(); gc.disable()

    # Free teacher models before saving (they're no longer needed)
    if teacher_models:
        del teacher_models
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # =========================================================================
    # Post-training: EMA blend evaluation
    # Save the final model weights so we can restore after each blend eval,
    # and compare blend vs final to pick the best checkpoint to save.
    # =========================================================================
    final_weights = {name: p.data.clone() for name, p in orig_model.named_parameters()}
    best_weights = final_weights  # start with final as best
    best_val_loss = min_val_loss
    best_val_bpb = min_val_bpb

    for ema in ema_trackers:
        print0(f"\n  [model {model_idx+1}] --- Evaluating EMA blend (decay={ema.decay}, {ema.num_updates} updates) ---")
        # Evaluate the known-best blend ratio: 0.6*final + 0.4*EMA
        alpha = 0.6
        blended_weights = {
            name: (
                alpha * final_weights[name]
                + (1 - alpha) * ema.shadow[name].to(final_weights[name].device, dtype=final_weights[name].dtype)
            )
            for name in final_weights
        }
        load_state_dict_into_model(orig_model, blended_weights)
        blend_model = orig_model if args.no_compile else torch.compile(orig_model, dynamic=False)
        blend_model.eval()
        _val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")
        val_loader = DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN, device=device, seed=0, quiet=True)
        with autocast_ctx:
            blend_bpb, blend_loss = evaluate_bpb(blend_model, val_loader, eval_steps, token_bytes)
        print0(f"  [model {model_idx+1}] Blend({alpha:.1f}*final+{1-alpha:.1f}*EMA {ema.decay}): Val BPB: {blend_bpb:.6f} | Val Loss: {blend_loss:.6f}")
        wandb_run.log({
            f"model_{model_idx+1}/ema_blend_bpb": blend_bpb,
            f"model_{model_idx+1}/ema_blend_loss": blend_loss,
            f"model_{model_idx+1}/ema_decay": ema.decay,
        })
        if blend_loss < best_val_loss:
            best_val_loss = blend_loss
            best_val_bpb = blend_bpb
            best_weights = blended_weights
            print0(f"  [model {model_idx+1}] ** New best! (blend {alpha:.1f}/{1-alpha:.1f} with EMA {ema.decay})")
        # Restore final weights before evaluating the next EMA candidate
        load_state_dict_into_model(orig_model, final_weights)

    # Load the best weights into orig_model for checkpointing
    if best_weights is not final_weights:
        print0(f"  [model {model_idx+1}] Saving EMA-blended weights to checkpoint (val_loss={best_val_loss:.6f})")
        load_state_dict_into_model(orig_model, best_weights)
    else:
        print0(f"  [model {model_idx+1}] Saving final weights to checkpoint (val_loss={best_val_loss:.6f})")

    # Save checkpoint (uncompiled model state_dict — best of final vs EMA blend)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_{model_idx}.pt")
    if int(os.environ.get('RANK', 0)) == 0:
        torch.save(orig_model.state_dict(), checkpoint_path)
    if ddp:
        dist.barrier()

    print0(f"  [model {model_idx+1}] Done. Best Val BPB: {best_val_bpb:.6f} | Best Val Loss: {best_val_loss:.6f}")
    print0(f"  Checkpoint saved to {checkpoint_path}")

    # Cleanup
    del model, orig_model, compiled_model, optimizer, train_loader
    gc.enable()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return checkpoint_path, best_val_bpb, best_val_loss


# =============================================================================
# Synchronized ensemble training (our additions)
# =============================================================================

def train_ensemble_sync(seeds, device, config, autocast_ctx, token_bytes,
                        wandb_run, ddp, ddp_world_size, checkpoint_dir,
                        num_epochs, ensemble_type, ensemble_mode):
    """Train N models in synchronized epoch-by-epoch fashion.

    Each epoch: train each model for 1 epoch, then evaluate all models individually
    and as an ensemble. Per-epoch checkpoints saved for every model.

    Logs to wandb:
      Per-step:  model_{i}/train_loss_raw, model_{i}/epoch, model_{i}/epoch_step,
                 model_{i}/tokens_seen, step_global
      Per-epoch: model_{i}/val_loss, model_{i}/val_bpb,
                 model_{i}/epoch_mean_train_loss,
                 ens/val_loss, ens/val_bpb, ens/num_models, epoch
    """
    N = len(seeds)
    master_process = (int(os.environ.get('RANK', 0)) == 0)
    synchronize = torch.cuda.synchronize if device.type == "cuda" else lambda: None

    print0(f"\n{'='*60}")
    print0(f"Synchronized ensemble training: {N} models × {num_epochs} epochs")
    print0(f"  ensemble_type={ensemble_type}, ensemble_mode={ensemble_mode}")
    print0(f"{'='*60}")

    _train_path = args.input_bin if args.input_bin else os.path.join(DATA_DIR, "fineweb_train.pt")
    _val_path = args.input_val_bin if args.input_val_bin else os.path.join(DATA_DIR, "fineweb_val.pt")

    # Build all models, optimizers, data loaders
    models = []           # orig (uncompiled) model references
    compiled_models = []  # compile-wrapped (or same as orig if --no-compile)
    optimizers = []
    loaders = []
    for i, seed in enumerate(seeds):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.device("meta"):
            m = GPT(config)
        m.to_empty(device=device)
        m.init_weights()
        models.append(m)
        compiled_models.append(m if args.no_compile else torch.compile(m, dynamic=False))
        optimizers.append(m.setup_optimizer())
        # Cross-epoch shuffle for both modes (differs across epochs):
        #   init         -> π_k   (data_seed shared=42, so all models see the same per-epoch permutation)
        #   init_shuffle -> π_{i,k} (data_seed per-model, independent per (model, epoch))
        data_seed = seed if ensemble_type == "init_shuffle" else 42
        loaders.append(DataLoader(_train_path, args.device_batch_size, MAX_SEQ_LEN,
                                  device=device, seed=data_seed, quiet=True,
                                  reshuffle_per_epoch=True,
                                  data_fraction=args.data_fraction))

    # Batch and iteration accounting (same for all models since same config)
    tokens_per_fwdbwd = args.device_batch_size * MAX_SEQ_LEN * ddp_world_size
    assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd
    tokens_per_epoch = loaders[0].total_tokens
    num_iterations = round(tokens_per_epoch * num_epochs / TOTAL_BATCH_SIZE)
    steps_per_epoch_optim = round(tokens_per_epoch / TOTAL_BATCH_SIZE)  # optimizer steps per epoch
    print0(f"  grad_accum_steps={grad_accum_steps}, steps_per_epoch_optim={steps_per_epoch_optim}, num_iterations={num_iterations}")
    print0(f"  tokens_per_epoch={tokens_per_epoch:,}  (data_fraction={args.data_fraction})")

    eval_steps = EVAL_TOKENS // (args.device_batch_size * MAX_SEQ_LEN * ddp_world_size)
    ens_eval_B = 1
    ens_eval_steps = EVAL_TOKENS // (ens_eval_B * MAX_SEQ_LEN * ddp_world_size)

    # Save full training config (for reproducibility / later analysis)
    if int(os.environ.get('RANK', 0)) == 0:
        # Pull learning rates from a freshly-built optimizer's param groups
        lr_groups = []
        for g in optimizers[0].param_groups:
            lr_groups.append({
                "kind": g["kind"],
                "lr": g["lr"],
                "weight_decay": g.get("weight_decay"),
                "betas": list(g["betas"]) if "betas" in g else None,
                "momentum": g.get("momentum"),
                "ns_steps": g.get("ns_steps"),
                "num_params": sum(p.numel() for p in g["params"]),
            })

        total_params = sum(p.numel() for p in models[0].parameters())
        total_training_tokens = tokens_per_epoch * num_epochs

        train_config = {
            "model": {
                "n_layer": config.n_layer,
                "n_head": config.n_head,
                "n_embd": config.n_embd,
                "vocab_size": config.vocab_size,
                "sequence_len": config.sequence_len,
                "dropout": config.dropout,
                "completep": config.completep,
                "mup_base_width": config.mup_base_width,
                "total_params": total_params,
            },
            "optimizer": {
                "name": args.optimizer,
                "weight_decay": WEIGHT_DECAY,
                "lr_multiplier": args.lr_multiplier,
                "base_matrix_lr": args.matrix_lr,
                "base_scalar_lr": args.scalar_lr,
                "warmup_ratio": WARMUP_RATIO,
                "warmdown_ratio": WARMDOWN_RATIO,
                "final_lr_frac": FINAL_LR_FRAC,
                "param_groups": lr_groups,
            },
            "training": {
                "num_epochs": num_epochs,
                "num_iterations": num_iterations,
                "steps_per_epoch": steps_per_epoch_optim,
                "tokens_per_epoch": tokens_per_epoch,
                "total_training_tokens": total_training_tokens,
                "total_batch_size": TOTAL_BATCH_SIZE,
                "device_batch_size": args.device_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "max_seq_len": MAX_SEQ_LEN,
                "ddp_world_size": ddp_world_size,
                "data_fraction": args.data_fraction,
                "no_compile": args.no_compile,
            },
            "ensemble": {
                "type": ensemble_type,
                "mode": ensemble_mode,
                "num_models": N,
                "seeds": list(seeds),
            },
            "val": {
                "tokens": EVAL_TOKENS,
                "every_n_steps": args.val_every_n_steps,
                "individual_eval_B": args.device_batch_size,
                "ensemble_eval_B": ens_eval_B,
            },
            "data": {
                "train_path": _train_path,
                "val_path": _val_path,
            },
            "args": vars(args),
        }
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(train_config, f, indent=2, default=str)
        print0(f"  Saved training config to {checkpoint_dir}/config.json")

    # LR schedule (shared — all models are identical config)
    def get_lr_multiplier(it):
        warmup = round(WARMUP_RATIO * num_iterations)
        warmdown = round(WARMDOWN_RATIO * num_iterations)
        if it < warmup: return (it + 1) / warmup
        elif it <= num_iterations - warmdown: return 1.0
        else:
            progress = (num_iterations - it) / warmdown
            return progress + (1 - progress) * FINAL_LR_FRAC

    def get_muon_momentum(it):
        return (1 - min(it / 300, 1)) * 0.85 + min(it / 300, 1) * 0.95

    # Per-model optim-step counters (each model has its own training progress)
    per_model_step = [0] * N
    step_global = 0

    # EMA smoothing per model
    smooth_train_loss = [0.0] * N

    individual_results = []   # [{"model": i, "seed": s, "per_epoch": [{"epoch": e, "val_loss": .., "val_bpb": ..}, ...]}, ...]
    ensemble_results = []     # [{"epoch": e, "val_loss": .., "val_bpb": ..}, ...]
    dupe_active = False
    dupe_start_epoch = math.ceil(args.dupe_fraction * num_epochs) + 1

    # Helper: individual val eval for ONE model at current training state
    def _run_individual_val(i, epoch):
        m = compiled_models[i]
        m.eval()
        vl_loader = DataLoader(_val_path, args.device_batch_size, MAX_SEQ_LEN,
                               device=device, seed=0, quiet=True)
        with autocast_ctx:
            vbpb, vloss = evaluate_bpb(m, vl_loader, eval_steps, token_bytes)
        wandb_run.log({
            f"model_{i+1}/val_bpb": vbpb,
            f"model_{i+1}/val_loss": vloss,
            f"model_{i+1}/epoch": epoch,
            f"model_{i+1}/step": per_model_step[i],
        }, commit=True)
        m.train()
        return vbpb, vloss

    # Helper: ensemble val eval (loads all N models together)
    def _run_ensemble_val(epoch):
        ens_loader = DataLoader(_val_path, ens_eval_B, MAX_SEQ_LEN,
                                device=device, seed=0, quiet=True)
        ebpb, eloss = evaluate_ensemble_in_memory(
            models, ens_loader, ens_eval_steps, token_bytes,
            device, autocast_ctx, mode=ensemble_mode)
        wandb_run.log({
            "ens/val_bpb": ebpb,
            "ens/val_loss": eloss,
            "ens/num_models": N,
            "ens/epoch": epoch,
        }, commit=True)
        return ebpb, eloss

    val_every_n = args.val_every_n_steps  # 0 = only at epoch boundaries

    for epoch in range(1, num_epochs + 1):
        print0(f"\n{'='*60}")
        print0(f"Epoch {epoch}/{num_epochs}")
        print0(f"{'='*60}")

        # Activate dupe layers (shared across all models) at the configured epoch
        if not dupe_active and epoch >= dupe_start_epoch:
            print0(f"  Enabling dupe-layers {args.dupe_layers_start}-{args.dupe_layers_end} at epoch {epoch}")
            for i in range(N):
                models[i].set_dupe_layers(args.dupe_layers_start, args.dupe_layers_end)
                compiled_models[i] = models[i] if args.no_compile else torch.compile(models[i], dynamic=False)
            dupe_active = True

        # ===== Sequential training: model i trains full epoch, then model i+1 =====
        # Per-step individual val eval happens inline. Ensemble val is only at epoch boundary.
        for i in range(N):
            m = compiled_models[i]
            opt = optimizers[i]
            loader = loaders[i]

            m.train()
            epoch_loss_sum = 0.0
            epoch_loss_count = 0

            for local_opt_step in range(steps_per_epoch_optim):
                synchronize()
                t0 = time.time()

                # Gradient accumulation
                for ga in range(grad_accum_steps):
                    x, y, _epoch = next(loader)
                    with autocast_ctx:
                        loss = m(x, y)
                    (loss / grad_accum_steps).backward()
                    train_loss_f = loss.item()
                    epoch_loss_sum += train_loss_f
                    epoch_loss_count += 1

                # Optimizer step
                global_it = per_model_step[i]
                lrm = get_lr_multiplier(global_it)
                for g in opt.param_groups:
                    g["lr"] = g["initial_lr"] * lrm
                    if g['kind'] == 'muon':
                        g["momentum"] = get_muon_momentum(global_it)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']], max_norm=1.0)
                opt.step()
                m.zero_grad(set_to_none=True)
                synchronize()
                dt = time.time() - t0
                toks_per_sec = TOTAL_BATCH_SIZE / dt
                per_model_step[i] += 1
                step_global += 1

                # EMA smoothing
                ema_beta = 0.9
                smooth_train_loss[i] = ema_beta * smooth_train_loss[i] + (1 - ema_beta) * train_loss_f
                debiased = smooth_train_loss[i] / (1 - ema_beta ** (local_opt_step + 1))

                if (local_opt_step + 1) % 50 == 0 or local_opt_step == 0:
                    print0(f"  [epoch {epoch}] [model {i+1}] step {local_opt_step+1}/{steps_per_epoch_optim} "
                           f"| loss: {debiased:.4f} | {toks_per_sec:.0f} tok/s")

                # Per-model training metrics
                wandb_run.log({
                    f"model_{i+1}/train_loss_raw": train_loss_f,
                    f"model_{i+1}/train_loss": debiased,
                    f"model_{i+1}/epoch": epoch,
                    f"model_{i+1}/epoch_step": local_opt_step + 1,
                    f"model_{i+1}/step": per_model_step[i],
                    f"model_{i+1}/tokens_seen": (epoch - 1) * tokens_per_epoch + (local_opt_step + 1) * TOTAL_BATCH_SIZE,
                }, commit=True)

                # Per-N-step individual val eval (cheap, no model-switching)
                if val_every_n > 0 and (local_opt_step + 1) % val_every_n == 0:
                    vbpb, vloss = _run_individual_val(i, epoch)
                    print0(f"    [model {i+1} val @ step {per_model_step[i]}] val_loss={vloss:.4f}")

            # Reset EMA at this model's epoch boundary
            smooth_train_loss[i] = 0.0
            mean_train_loss = epoch_loss_sum / max(1, epoch_loss_count)
            print0(f"  [model {i+1}] epoch {epoch} mean_train_loss={mean_train_loss:.4f}")
            wandb_run.log({
                f"model_{i+1}/epoch_mean_train_loss": mean_train_loss,
                f"model_{i+1}/epoch": epoch,
            }, commit=True)

        # ===== Epoch-boundary: ensemble val eval (all models now at same epoch) =====
        ens_bpb, ens_loss = _run_ensemble_val(epoch)
        print0(f"  [ensemble] epoch {epoch} val_bpb={ens_bpb:.6f} val_loss={ens_loss:.6f}")
        ensemble_results.append({"epoch": epoch, "val_bpb": ens_bpb, "val_loss": ens_loss})

        # ===== Save per-epoch checkpoints =====
        if master_process:
            for i in range(N):
                ckpt_path = os.path.join(checkpoint_dir, f"model_{i}_epoch_{epoch}.pt")
                torch.save(models[i].state_dict(), ckpt_path)
            # Save progress summary
            progress = {
                "seeds": seeds,
                "ensemble_type": ensemble_type,
                "ensemble_mode": ensemble_mode,
                "ensemble_results": ensemble_results,
                "epochs_completed": epoch,
                "num_epochs": num_epochs,
            }
            with open(os.path.join(checkpoint_dir, "progress.json"), "w") as f:
                json.dump(progress, f, indent=2)
        if ddp:
            dist.barrier()

        # Switch models back to train mode for next epoch
        for m in compiled_models:
            m.train()

    return ensemble_results


# =============================================================================
# Main: train ensemble
# =============================================================================

def main():
    total_start_time = time.time()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    master_process = ddp_rank == 0

    if ddp and torch.cuda.is_available():
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_type = device.type
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Flash Attention status
    if _fa3 is not None:
        print0("Using Flash Attention 3 (Hopper GPU detected)")
    elif _fa2 is not None:
        print0("Using Flash Attention 2 (FA3 not available)")
    else:
        print0("Using PyTorch SDPA fallback (no FA3/FA2 available; sliding window attention disabled)")

    # wandb + run_id
    if args.resume:
        run_id = args.resume
        checkpoint_dir = os.path.join(args.checkpoint_base, run_id)
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Resume directory does not exist: {checkpoint_dir}")
        print0(f"Resuming run: {run_id}")
    else:
        run_id = time.strftime('%Y%m%d_%H%M%S')
        checkpoint_dir = os.path.join(args.checkpoint_base, run_id)
        print0(f"New run: {run_id}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    run_name = args.run if args.run else f"ensemble_{run_id}"
    _wandb_kwargs = {"project": "slowrun", "name": run_name,
                     "config": {"optimizer": args.optimizer, "completep": args.completep,
                                "ensemble_type": args.ensemble_type, "ensemble_mode": args.ensemble_mode,
                                "mup_base_width": args.mup_base_width,
                                "n_layer": DEPTH, "n_embd": N_EMBD, "n_head": N_HEAD,
                                "num_models": args.num_models, "num_epochs": args.num_epochs,
                                "dropout": args.dropout, "weight_decay": WEIGHT_DECAY}}
    if args.wandb_group:
        _wandb_kwargs["group"] = args.wandb_group
    wandb_run = DummyWandb() if not master_process else wandb.init(**_wandb_kwargs)

    # Tokenizer + token_bytes
    encoder = tiktoken.get_encoding("gpt2")
    vocab_size = encoder.n_vocab
    eot_id = encoder._special_tokens['<|endoftext|>']
    token_bytes_list = []
    for i in range(vocab_size):
        if i == eot_id:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(encoder.decode_single_token_bytes(i)))
    token_bytes = torch.tensor(token_bytes_list, dtype=torch.int32, device=device)

    config = GPTConfig(vocab_size=vocab_size, dropout=args.dropout,
                       completep=args.completep, mup_base_width=args.mup_base_width,
                       optimizer=args.optimizer)

    # Print config
    print0(f"\n{'='*60}")
    print0(f"Ensemble Training: {args.num_models} models")
    print0(f"{'='*60}")
    print0(f"  run_id={run_id}  (resume with: --resume {run_id})")
    print0(f"  n_layer={DEPTH}, n_embd={N_EMBD}, n_head={N_HEAD}")
    print0(f"  num_epochs={args.num_epochs}, dropout={args.dropout}")
    dupe_start = math.ceil(args.dupe_fraction * args.num_epochs) + 1
    if dupe_start <= args.num_epochs:
        print0(f"  dupe_layers={args.dupe_layers_start}-{args.dupe_layers_end}, activates at epoch {dupe_start}")
    else:
        print0(f"  dupe_layers: disabled (dupe_fraction={args.dupe_fraction})")
    print0(f"  optimizer={args.optimizer}, ensemble_type={args.ensemble_type}, ensemble_mode={args.ensemble_mode}")
    if args.completep:
        print0(f"  completep=True, mup_base_width={args.mup_base_width}, depth_scale={1.0/DEPTH:.4f}, output_mult={args.mup_base_width/N_EMBD:.4f}")
    print0(f"  checkpoint_dir={checkpoint_dir}")
    print0(f"{'='*60}")

    # Seeds for each model
    seeds = [42 + i for i in range(args.num_models)]

    # Resume is not supported in synchronized training mode — progress checkpoints are per-epoch
    if args.resume:
        print0("WARNING: --resume is not fully supported in synchronized training mode; starting from epoch 1")
    if args.distill_alpha != 0.0 and not args.no_distill:
        print0("NOTE: distillation ignored in synchronized training mode; --distill-alpha has no effect")

    # Synchronized ensemble training: all N models train epoch-by-epoch together
    ensemble_results = train_ensemble_sync(
        seeds=seeds,
        device=device,
        config=config,
        autocast_ctx=autocast_ctx,
        token_bytes=token_bytes,
        wandb_run=wandb_run,
        ddp=ddp,
        ddp_world_size=ddp_world_size,
        checkpoint_dir=checkpoint_dir,
        num_epochs=args.num_epochs,
        ensemble_type=args.ensemble_type,
        ensemble_mode=args.ensemble_mode,
    )

    # Final summary
    print0(f"\n{'='*60}")
    print0(f"Ensemble Training Complete")
    print0(f"{'='*60}")
    print0(f"\nPer-epoch ensemble results ({args.ensemble_mode} averaging, {args.num_models} models):")
    for r in ensemble_results:
        print0(f"  Epoch {r['epoch']:3d}: BPB={r['val_bpb']:.6f}, Loss={r['val_loss']:.6f}")
    if ensemble_results:
        final = ensemble_results[-1]
        print0(f"\n*** Final ensemble (epoch {final['epoch']}): BPB={final['val_bpb']:.6f} | Val Loss={final['val_loss']:.6f} ***")

    # Save results
    if args.save_result and master_process:
        result = {
            "seeds": seeds,
            "ensemble_type": args.ensemble_type,
            "ensemble_mode": args.ensemble_mode,
            "num_epochs": args.num_epochs,
            "ensemble_results": ensemble_results,
        }
        with open(args.save_result, "w") as f:
            json.dump(result, f, indent=2)
        print0(f"Results saved to {args.save_result}")

    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print0(f"\nTotal time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")

    wandb_run.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
