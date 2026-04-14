import os
import sys
import time
import argparse
import json
import traceback
from typing import Tuple, Dict, Optional

import torch
import torch_npu
from safetensors import safe_open


# ==================== 路径 ====================
sys.path.append("/docker/w00888862/llm_quant_test/llm_quant")
sys.path.append("/docker/w00888862/llm_quant_test/llm_quant/quant_cy/base/QFuncs")


# ==================== 配置 ====================
NUM_EXPERTS = 64
LAYER_START = 1
LAYER_END = 25

EPOCHS = 300
LR = 3e-3
WEIGHT_DECAY = 0.0

TAU_START = 2.0
TAU_END = 0.05
SOFTSORT_METRIC = "l1"          # "l1" 或 "l2"

MXFP4_GROUP_SIZE = 32

# 早停
EARLY_STOP_PATIENCE = 40
EARLY_STOP_REL_DELTA = 1e-4     # 相对提升阈值，比 1e-10 更合理
EMA_BETA_FOR_EARLYSTOP = 0.9    # 对 hard_eval_loss 做 EMA，减少抖动

# 正则
DOUBLY_STOCHASTIC_REG = 1e-2    # 行列和接近 1

# 数值稳定
EPS = 1e-8
SINKHORN_ITERS = 10

ACT_SAMPLES_PATH = "/docker/w00888862/llm_quant_test/llm_quant/ptq/smoothquant/act_scales/deepseek-v2-lite_acts.pt"
WEIGHTS_PATH = "/docker/models/DeepSeek-V2-Lite/merged_model.safetensors"

OUTPUT_ROOT = "PermuteDown_BatchScore_HardForward_SoftBackward_STE_MXFP4_v2"


# ==================== 调度 ====================
def get_tau(epoch: int, total_epochs: int, tau_start: float, tau_end: float) -> float:
    """
    指数退火比线性更适合排序松弛。
    """
    if total_epochs <= 1:
        return tau_end
    ratio = epoch / (total_epochs - 1)
    return tau_start * ((tau_end / tau_start) ** ratio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train permutation scores for MXFP4 quantization.")
    parser.add_argument("--act-samples-path", type=str, default=ACT_SAMPLES_PATH)
    parser.add_argument("--weights-path", type=str, default=WEIGHTS_PATH)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--num-experts", type=int, default=NUM_EXPERTS)
    parser.add_argument("--layer-start", type=int, default=LAYER_START)
    parser.add_argument("--layer-end", type=int, default=LAYER_END)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--tau-start", type=float, default=TAU_START)
    parser.add_argument("--tau-end", type=float, default=TAU_END)
    parser.add_argument("--softsort-metric", type=str, default=SOFTSORT_METRIC, choices=["l1", "l2"])
    parser.add_argument("--mxfp4-group-size", type=int, default=MXFP4_GROUP_SIZE)
    parser.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    parser.add_argument("--early-stop-rel-delta", type=float, default=EARLY_STOP_REL_DELTA)
    parser.add_argument("--ema-beta-for-earlystop", type=float, default=EMA_BETA_FOR_EARLYSTOP)
    parser.add_argument("--doubly-stochastic-reg", type=float, default=DOUBLY_STOCHASTIC_REG)
    parser.add_argument("--sinkhorn-iters", type=int, default=SINKHORN_ITERS)
    return parser.parse_args()


def resolve_output_root(output_root: Optional[str]) -> str:
    if output_root:
        return output_root
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(OUTPUT_ROOT, timestamp)


# ==================== MXFP4 fake quant ====================
def dq_MXFP4(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    MXFP4 dequant (Fake quant)

    支持:
      x: [D]
      x: [B, D]
      x: [N, D]
    只要求最后一维是待分组量化的维度
    """
    orig_shape = x.shape
    last_dim = orig_shape[-1]

    assert last_dim % group_size == 0, (
        f"last_dim={last_dim} 不能整除 group_size={group_size}"
    )

    x_2d = x.reshape(-1, last_dim)                          # [N, D]
    x_3d = x_2d.reshape(x_2d.shape[0], -1, group_size)     # [N, G, group_size]

    # block scale
    amax = torch.max(torch.abs(x_3d), dim=2, keepdim=True)[0]
    scale = amax * 8.0 / 7.0
    scale = torch.exp2(torch.floor(torch.log2(scale + 1e-7))).clamp_min(1e-8)
    scale = scale.detach()

    x_scaled = x_3d / scale
    x_scaled = torch.clamp(x_scaled, -1.5, 1.5)

    # secondary scale
    exp = torch.floor(torch.log2(torch.abs(x_scaled) + 1e-7))
    exp = torch.clamp(exp, -2, 0)
    sec_scale = torch.exp2(exp - 1).clamp_min(1e-8)
    sec_scale = sec_scale.detach()

    x_q = torch.round(x_scaled / sec_scale) * sec_scale
    x_q = x_q * scale
    x_q = x_q.reshape(-1, last_dim)
    x_q = x_q.reshape(orig_shape)

    return x_q


def Qh(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    显式 STE:
      forward  用量化值
      backward 当 identity
    """
    x_q = dq_MXFP4(x, group_size=group_size)
    return x + (x_q - x).detach()


# ==================== permutation / Sinkhorn ====================
def normalize_score_for_sort(score: torch.Tensor) -> torch.Tensor:
    """
    排序变量做标准化，避免 score 绝对尺度与 tau 强耦合。
    """
    s = score.float()
    s = s - s.mean()
    s = s / (s.std() + 1e-6)
    return s


def build_soft_cost_matrix(scores: torch.Tensor, metric: str = "l1") -> torch.Tensor:
    """
    构造排序 cost matrix:
      行 = 排序后位置对应的目标值
      列 = 原始元素
    """
    s_sorted, _ = torch.sort(scores, descending=False)

    if metric == "l1":
        cost = torch.abs(s_sorted[:, None] - scores[None, :])
    elif metric == "l2":
        cost = (s_sorted[:, None] - scores[None, :]) ** 2
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return cost


def sinkhorn_normalization(log_alpha: torch.Tensor, n_iters: int = 10) -> torch.Tensor:
    """
    对 exp(log_alpha) 做 Sinkhorn，使之近似双随机矩阵。
    输入 / 输出形状: [D, D]
    """
    P = torch.exp(log_alpha)

    for _ in range(n_iters):
        P = P / (P.sum(dim=1, keepdim=True) + EPS)
        P = P / (P.sum(dim=0, keepdim=True) + EPS)

    return P


def softsort_perm_matrix_sinkhorn(
    scores: torch.Tensor,
    tau: float = 1.0,
    metric: str = "l1",
    sinkhorn_iters: int = 10,
) -> torch.Tensor:
    """
    用 cost + Sinkhorn 得到更像 permutation 的 soft matrix。
    """
    scores = normalize_score_for_sort(scores)
    cost = build_soft_cost_matrix(scores, metric=metric)
    log_alpha = -cost / max(tau, 1e-6)
    P_soft = sinkhorn_normalization(log_alpha, n_iters=sinkhorn_iters)
    return P_soft


def hard_perm_indices_from_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    perm_hard[r] = old_index
    """
    scores = normalize_score_for_sort(scores)
    return torch.argsort(scores, descending=False)


def hard_perm_matrix_from_indices(perm: torch.Tensor) -> torch.Tensor:
    """
    根据 perm 构造 P_hard [D, D]

    语义:
      新位置 r 取旧位置 perm[r]

    则:
      P_hard[r, perm[r]] = 1
      x_perm = x @ P_hard.T
    """
    D = perm.numel()
    P_hard = torch.zeros(D, D, device=perm.device, dtype=torch.float32)
    rows = torch.arange(D, device=perm.device)
    P_hard[rows, perm] = 1.0
    return P_hard


def ste_perm_matrix_from_scores(
    scores: torch.Tensor,
    tau: float = 1.0,
    metric: str = "l1",
    sinkhorn_iters: int = 10,
):
    """
    前向 hard，反向 soft 的 permutation STE

    返回:
      perm_hard: [D]
      P_soft:    [D, D]
      P_hard:    [D, D]
      P_ste:     [D, D]
    """
    perm_hard = hard_perm_indices_from_scores(scores)
    P_soft = softsort_perm_matrix_sinkhorn(
        scores, tau=tau, metric=metric, sinkhorn_iters=sinkhorn_iters
    )
    P_hard = hard_perm_matrix_from_indices(perm_hard)
    P_ste = P_hard - P_soft.detach() + P_soft
    return perm_hard, P_soft, P_hard, P_ste


def apply_hard_perm_batch(
    x: torch.Tensor,
    w: torch.Tensor,
    perm: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x:    [B, D]
    w:    [O, D]
    perm: [D]

    语义:
      新位置 r 取旧位置 perm[r]
    """
    x_perm = x[:, perm]   # [B, D]
    w_perm = w[:, perm]   # [O, D]
    return x_perm, w_perm


def apply_perm_matrix_batch(
    x: torch.Tensor,
    w: torch.Tensor,
    P: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: [B, D]
    w: [O, D]
    P: [D, D]

    约定:
      x_perm[b, r] = sum_i x[b, i] * P[r, i]
      w_perm[o, r] = sum_i w[o, i] * P[r, i]

    等价矩阵形式:
      x_perm = x @ P.T
      w_perm = w @ P.T
    """
    x_perm = x @ P.T
    w_perm = w @ P.T
    return x_perm, w_perm


# ==================== loss / regularization ====================
def matmul_mse_full(
    x_q: torch.Tensor,      # [B, D]
    w_q: torch.Tensor,      # [O, D]
    y_ref: torch.Tensor,    # [B, O]
) -> torch.Tensor:
    """
    直接整块计算 matmul MSE。
    """
    y_q = torch.matmul(x_q, w_q.T)   # [B, O]
    diff = y_q - y_ref
    return (diff * diff).mean()


def compute_quant_loss_hard(
    x: torch.Tensor,
    w: torch.Tensor,
    y_ref: torch.Tensor,
    perm: torch.Tensor,
    group_size: int,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    用 hard permutation 后的 x, w 做 MXFP4 fake quant
    """
    x_perm, w_perm = apply_hard_perm_batch(x, w, perm)

    x_q = Qh(x_perm, group_size=group_size)
    w_q = Qh(w_perm, group_size=group_size)

    loss_recon = matmul_mse_full(x_q, w_q, y_ref)

    if not return_aux:
        return loss_recon, None

    aux = {
        "x_perm": x_perm.detach(),
        "w_perm": w_perm.detach(),
        "x_q": x_q.detach(),
        "w_q": w_q.detach(),
        "loss_recon": loss_recon.detach(),
    }
    return loss_recon, aux


def compute_quant_loss_perm_matrix(
    x: torch.Tensor,
    w: torch.Tensor,
    y_ref: torch.Tensor,
    P: torch.Tensor,
    group_size: int,
    return_aux: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    用 permutation matrix（可为 STE matrix）做训练
    """
    x_perm, w_perm = apply_perm_matrix_batch(x, w, P)

    x_q = Qh(x_perm, group_size=group_size)
    w_q = Qh(w_perm, group_size=group_size)

    loss_recon = matmul_mse_full(x_q, w_q, y_ref)

    if not return_aux:
        return loss_recon, None

    aux = {
        "x_perm": x_perm.detach(),
        "w_perm": w_perm.detach(),
        "x_q": x_q.detach(),
        "w_q": w_q.detach(),
        "loss_recon": loss_recon.detach(),
    }
    return loss_recon, aux


def doubly_stochastic_regularization(P: torch.Tensor) -> torch.Tensor:
    row_err = ((P.sum(dim=1) - 1.0) ** 2).mean()
    col_err = ((P.sum(dim=0) - 1.0) ** 2).mean()
    return row_err + col_err


def permutation_distance_to_identity(perm: torch.Tensor) -> float:
    D = perm.numel()
    identity = torch.arange(D, device=perm.device)
    return (perm != identity).float().mean().item()


def mean_displacement_to_identity(perm: torch.Tensor) -> float:
    D = perm.numel()
    identity = torch.arange(D, device=perm.device)
    return float((perm - identity).abs().float().mean().item())


def max_displacement_to_identity(perm: torch.Tensor) -> int:
    D = perm.numel()
    identity = torch.arange(D, device=perm.device)
    return int((perm - identity).abs().max().item())


def matrix_entropy_rows(P: torch.Tensor, eps: float = 1e-12) -> float:
    P_clamped = P.clamp_min(eps)
    ent = -(P_clamped * torch.log(P_clamped)).sum(dim=1).mean()
    return float(ent.detach().cpu())


def matrix_avg_max_prob(P: torch.Tensor) -> float:
    return float(P.max(dim=1).values.mean().detach().cpu())


def matrix_row_col_error(P: torch.Tensor) -> Tuple[float, float]:
    row_err = float(((P.sum(dim=1) - 1.0) ** 2).mean().detach().cpu())
    col_err = float(((P.sum(dim=0) - 1.0) ** 2).mean().detach().cpu())
    return row_err, col_err


def sanity_check_permutation_equivariance(
    x: torch.Tensor,
    w: torch.Tensor,
    perm: torch.Tensor,
) -> float:
    """
    不量化时，若对 x 和 w 做一致的输入通道重排，输出应严格相同
    """
    x_perm, w_perm = apply_hard_perm_batch(x, w, perm)
    y0 = x @ w.T
    y1 = x_perm @ w_perm.T
    return float((y0 - y1).abs().max().detach().cpu())


# ==================== 模型 ====================
class ScorePerm1D(torch.nn.Module):
    """
    单个 (layer, expert) 对应一个长度为 D 的一维 score
    """
    def __init__(self, dim: int, init_mode: str = "identity_small_noise"):
        super().__init__()

        if init_mode == "zeros":
            init = torch.zeros(dim, dtype=torch.float32)
        elif init_mode == "randn":
            init = 0.01 * torch.randn(dim, dtype=torch.float32)
        elif init_mode == "identity_small_noise":
            base = torch.arange(dim, dtype=torch.float32)
            base = base / max(dim - 1, 1)
            init = base + 0.001 * torch.randn(dim, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

        self.score = torch.nn.Parameter(init)

    def forward(self, tau: float, metric: str = "l1", sinkhorn_iters: int = 10):
        return ste_perm_matrix_from_scores(
            self.score, tau=tau, metric=metric, sinkhorn_iters=sinkhorn_iters
        )


# ==================== 数据加载 ====================
def load_single_down(
    layer_id: int,
    expert_id: int,
    act_samples_full: Dict[str, torch.Tensor],
    weight_reader,
):
    """
    加载一个 layer + expert 的 down_proj

    期望:
      x: [B, D]
      w: [O, D]   # 保持原始权重布局，不转置
    """
    key = f"model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj"
    w_key = key + ".weight"

    if key not in act_samples_full:
        raise KeyError(f"{key} not found in act samples")

    x = act_samples_full[key].float()

    if x.dim() != 2:
        raise ValueError(f"{key}: expected x shape [B, D], but got {tuple(x.shape)}")

    w = weight_reader.get_tensor(w_key).float()   # [O, D]

    if x.shape[1] != w.shape[1]:
        raise ValueError(
            f"{key}: x.shape={tuple(x.shape)}, w.shape={tuple(w.shape)} not aligned on input dim D"
        )

    return key, x, w


# ==================== 局部 hard refinement ====================
@torch.no_grad()
def local_swap_refine(
    x: torch.Tensor,
    w: torch.Tensor,
    y_ref: torch.Tensor,
    perm_init: torch.Tensor,
    group_size: int,
    n_steps: int = 64,
) -> Tuple[torch.Tensor, float]:
    """
    对当前 best_perm 做少量 pair-swap hill climbing。
    成本可控，但通常能捡到一点 hard 目标收益。
    """
    perm = perm_init.clone()
    best_loss, _ = compute_quant_loss_hard(
        x, w, y_ref, perm, group_size=group_size, return_aux=False
    )
    best_loss_val = float(best_loss.detach().cpu())

    D = perm.numel()
    if D < 2:
        return perm, best_loss_val

    for _ in range(n_steps):
        i = torch.randint(0, D, (1,), device=perm.device).item()
        j = torch.randint(0, D, (1,), device=perm.device).item()
        if i == j:
            continue

        cand = perm.clone()
        cand[i], cand[j] = cand[j].clone(), cand[i].clone()

        cand_loss, _ = compute_quant_loss_hard(
            x, w, y_ref, cand, group_size=group_size, return_aux=False
        )
        cand_loss_val = float(cand_loss.detach().cpu())

        if cand_loss_val < best_loss_val:
            perm = cand
            best_loss_val = cand_loss_val

    return perm, best_loss_val


def save_expert_result(
    out_dir: str,
    layer_id: int,
    expert_id: int,
    save_obj: Dict,
    summary: Dict,
) -> None:
    save_path = os.path.join(out_dir, f"layer{layer_id:02d}_expert{expert_id:02d}.pt")
    torch.save(save_obj, save_path)
    print(f"saved: {save_path}")

    summary_path = os.path.join(out_dir, f"layer{layer_id:02d}_expert{expert_id:02d}.summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_optimizer_scheduler(
    model: torch.nn.Module,
    cfg: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.1
    )
    return optimizer, scheduler


def run_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    x: torch.Tensor,
    w: torch.Tensor,
    y_ref: torch.Tensor,
    cfg: argparse.Namespace,
    tau: float,
    metric: str,
) -> Tuple[float, torch.Tensor]:
    optimizer.zero_grad()

    _, P_soft_train, _, P_ste_train = model(
        tau=tau, metric=metric, sinkhorn_iters=cfg.sinkhorn_iters
    )

    loss, _ = compute_quant_loss_perm_matrix(
        x, w, y_ref, P_ste_train, group_size=cfg.mxfp4_group_size, return_aux=False
    )

    reg_ds = cfg.doubly_stochastic_reg * doubly_stochastic_regularization(P_soft_train)
    total_loss = loss + reg_ds
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    scheduler.step()

    return float(loss.detach().cpu()), P_soft_train.detach()


def evaluate_current_perm(
    model: torch.nn.Module,
    x: torch.Tensor,
    w: torch.Tensor,
    y_ref: torch.Tensor,
    cfg: argparse.Namespace,
    tau: float,
    metric: str,
    hard_eval_ema: Optional[float],
) -> Dict[str, float]:
    with torch.no_grad():
        perm_hard_eval, P_soft_eval, _, _ = model(
            tau=tau, metric=metric, sinkhorn_iters=cfg.sinkhorn_iters
        )
        hard_eval_loss, _ = compute_quant_loss_hard(
            x, w, y_ref, perm_hard_eval, group_size=cfg.mxfp4_group_size, return_aux=False
        )
        hard_eval_loss_val = float(hard_eval_loss.cpu())

        if hard_eval_ema is None:
            hard_eval_ema = hard_eval_loss_val
        else:
            hard_eval_ema = cfg.ema_beta_for_earlystop * hard_eval_ema + \
                            (1.0 - cfg.ema_beta_for_earlystop) * hard_eval_loss_val

        changed_ratio = permutation_distance_to_identity(perm_hard_eval)
        mean_disp = mean_displacement_to_identity(perm_hard_eval)
        max_disp = max_displacement_to_identity(perm_hard_eval)
        soft_entropy = matrix_entropy_rows(P_soft_eval)
        soft_avg_maxprob = matrix_avg_max_prob(P_soft_eval)
        soft_row_err, soft_col_err = matrix_row_col_error(P_soft_eval)

    return {
        "perm_hard_eval": perm_hard_eval.detach().cpu().clone(),
        "hard_eval_loss_val": hard_eval_loss_val,
        "hard_eval_ema": float(hard_eval_ema),
        "changed_ratio": changed_ratio,
        "mean_disp": mean_disp,
        "max_disp": max_disp,
        "soft_entropy": soft_entropy,
        "soft_avg_maxprob": soft_avg_maxprob,
        "soft_row_err": soft_row_err,
        "soft_col_err": soft_col_err,
    }


# ==================== 训练单个专家 ====================
def train_one_expert(
    layer_id: int,
    expert_id: int,
    act_samples_full: Dict[str, torch.Tensor],
    weight_reader,
    device: torch.device,
    out_dir: str,
    cfg: argparse.Namespace,
    metric: str = "l1",
) -> None:
    key, x_cpu, w_cpu = load_single_down(
        layer_id=layer_id,
        expert_id=expert_id,
        act_samples_full=act_samples_full,
        weight_reader=weight_reader,
    )

    x = x_cpu.to(device, dtype=torch.float32)  # [B, D]
    w = w_cpu.to(device, dtype=torch.float32)  # [O, D]

    B, D = x.shape
    O = w.shape[0]

    assert D % cfg.mxfp4_group_size == 0, (
        f"{key}: hidden dim {D} 不能整除 MXFP4_GROUP_SIZE={cfg.mxfp4_group_size}"
    )

    # 预计算 reference output，避免每个 epoch 重复算
    with torch.no_grad():
        y_ref = torch.matmul(x, w.T)

    model = ScorePerm1D(D, init_mode="identity_small_noise").to(device)
    optimizer, scheduler = build_optimizer_scheduler(model, cfg)

    # sanity check: 不量化时 permutation 前后应等价
    with torch.no_grad():
        perm_rand = torch.randperm(D, device=device)
        equiv_err = sanity_check_permutation_equivariance(x, w, perm_rand)
        print(f"[Sanity] {key}: permutation no-quant max_abs_err = {equiv_err:.6e}")

    # identity baseline
    with torch.no_grad():
        perm_identity = torch.arange(D, device=device)
        baseline_loss, _ = compute_quant_loss_hard(
            x, w, y_ref, perm_identity, group_size=cfg.mxfp4_group_size, return_aux=False
        )
        baseline_loss = float(baseline_loss.cpu())

    best_loss = float("inf")
    best_ema = float("inf")
    best_perm = None
    best_score = None
    best_epoch = -1

    loss_history = []
    loss_ema_history = []
    tau_history = []
    lr_history = []
    changed_ratio_history = []
    mean_displacement_history = []
    max_displacement_history = []
    soft_entropy_history = []
    soft_avg_maxprob_history = []
    soft_row_error_history = []
    soft_col_error_history = []

    no_improve_steps = 0
    hard_eval_ema = None
    t0 = time.time()

    for epoch in range(cfg.epochs):
        tau = get_tau(epoch, cfg.epochs, cfg.tau_start, cfg.tau_end)
        train_loss, _ = run_train_step(
            model, optimizer, scheduler, x, w, y_ref, cfg, tau, metric
        )
        eval_stats = evaluate_current_perm(
            model, x, w, y_ref, cfg, tau, metric, hard_eval_ema
        )

        hard_eval_ema = eval_stats["hard_eval_ema"]
        hard_eval_loss_val = eval_stats["hard_eval_loss_val"]
        changed_ratio = eval_stats["changed_ratio"]
        mean_disp = eval_stats["mean_disp"]
        max_disp = int(eval_stats["max_disp"])
        soft_entropy = eval_stats["soft_entropy"]
        soft_avg_maxprob = eval_stats["soft_avg_maxprob"]
        soft_row_err = eval_stats["soft_row_err"]
        soft_col_err = eval_stats["soft_col_err"]

        improved = False
        if best_ema == float("inf"):
            improved = True
        else:
            threshold = best_ema * (1.0 - cfg.early_stop_rel_delta)
            if hard_eval_ema < threshold:
                improved = True

        if improved:
            best_ema = hard_eval_ema
            best_loss = hard_eval_loss_val
            best_perm = eval_stats["perm_hard_eval"]
            best_score = model.score.detach().cpu().clone()
            best_epoch = epoch
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        loss_history.append(hard_eval_loss_val)
        loss_ema_history.append(hard_eval_ema)
        tau_history.append(float(tau))
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        changed_ratio_history.append(changed_ratio)
        mean_displacement_history.append(mean_disp)
        max_displacement_history.append(max_disp)
        soft_entropy_history.append(soft_entropy)
        soft_avg_maxprob_history.append(soft_avg_maxprob)
        soft_row_error_history.append(soft_row_err)
        soft_col_error_history.append(soft_col_err)

        if epoch % 20 == 0 or epoch == cfg.epochs - 1:
            improve_ratio = (baseline_loss - hard_eval_loss_val) / max(abs(baseline_loss), 1e-12)
            print(
                f"[layer={layer_id:02d}, expert={expert_id:02d}] "
                f"B={B} D={D} O={O} "
                f"epoch={epoch:04d} "
                f"tau={tau:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} "
                f"train_loss={train_loss:.6e} "
                f"hard_eval_loss={hard_eval_loss_val:.6e} "
                f"hard_eval_ema={hard_eval_ema:.6e} "
                f"baseline={baseline_loss:.6e} "
                f"improve={improve_ratio:.4%} "
                f"changed_ratio={changed_ratio:.4f} "
                f"mean_disp={mean_disp:.2f} "
                f"max_disp={max_disp:d} "
                f"soft_entropy={soft_entropy:.4f} "
                f"soft_avg_maxprob={soft_avg_maxprob:.4f} "
                f"row_err={soft_row_err:.3e} "
                f"col_err={soft_col_err:.3e}"
            )

        if no_improve_steps >= cfg.early_stop_patience:
            print(
                f"[EarlyStop] layer={layer_id:02d}, expert={expert_id:02d}, "
                f"epoch={epoch:04d}, patience={cfg.early_stop_patience}"
            )
            break

    # 对 best_perm 做一点 cheap hard refinement
    if best_perm is None:
        print(f"[Warn] {key}: no best_perm captured during training, fallback to identity permutation.")
        best_perm = torch.arange(D, dtype=torch.long)
        best_score = model.score.detach().cpu().clone()
        best_epoch = -1
        best_ema = float("nan")
        best_loss, _ = compute_quant_loss_hard(
            x, w, y_ref, best_perm.to(device), group_size=cfg.mxfp4_group_size, return_aux=False
        )
        best_loss = float(best_loss.cpu())

    best_perm_device = best_perm.to(device)
    refined_perm, refined_loss = local_swap_refine(
        x, w, y_ref, best_perm_device, group_size=cfg.mxfp4_group_size, n_steps=64
    )
    refined_perm_cpu = refined_perm.detach().cpu().clone()

    final_best_loss = min(best_loss, refined_loss)
    final_best_perm = best_perm if best_loss <= refined_loss else refined_perm_cpu
    final_best_source = "train_best" if best_loss <= refined_loss else "local_swap_refine"

    improvement_vs_identity = (baseline_loss - final_best_loss) / max(abs(baseline_loss), 1e-12)

    save_obj = {
        "key": key,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "config": vars(cfg).copy(),
        "batch_size_tokens": B,
        "dim": D,
        "out_dim": O,
        "metric": metric,
        "epochs_config": cfg.epochs,
        "epochs_ran": len(loss_history),
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "doubly_stochastic_reg": cfg.doubly_stochastic_reg,
        "tau_start": cfg.tau_start,
        "tau_end": cfg.tau_end,
        "sinkhorn_iters": cfg.sinkhorn_iters,
        "mxfp4_group_size": cfg.mxfp4_group_size,
        "early_stop_patience": cfg.early_stop_patience,
        "early_stop_rel_delta": cfg.early_stop_rel_delta,
        "ema_beta_for_earlystop": cfg.ema_beta_for_earlystop,
        "baseline_identity_loss": baseline_loss,
        "best_ema_before_refine": best_ema,
        "best_loss_before_refine": best_loss,
        "best_epoch": best_epoch,
        "refined_loss": refined_loss,
        "final_best_loss": final_best_loss,
        "final_best_source": final_best_source,
        "improvement_vs_identity": improvement_vs_identity,
        "best_perm": final_best_perm,
        "best_score": best_score,
        "loss_history": loss_history,
        "loss_ema_history": loss_ema_history,
        "tau_history": tau_history,
        "lr_history": lr_history,
        "changed_ratio_history": changed_ratio_history,
        "mean_displacement_history": mean_displacement_history,
        "max_displacement_history": max_displacement_history,
        "soft_entropy_history": soft_entropy_history,
        "soft_avg_maxprob_history": soft_avg_maxprob_history,
        "soft_row_error_history": soft_row_error_history,
        "soft_col_error_history": soft_col_error_history,
        "time_sec": time.time() - t0,
    }

    summary = {
        "key": key,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "best_ema_before_refine": best_ema,
        "baseline_identity_loss": baseline_loss,
        "best_loss_before_refine": best_loss,
        "refined_loss": refined_loss,
        "final_best_loss": final_best_loss,
        "final_best_source": final_best_source,
        "best_epoch": best_epoch,
        "improvement_vs_identity": improvement_vs_identity,
        "epochs_ran": len(loss_history),
        "time_sec": time.time() - t0,
    }
    save_expert_result(out_dir, layer_id, expert_id, save_obj, summary)


# ==================== 主函数 ====================
def main() -> None:
    cfg = parse_args()
    cfg.output_root = resolve_output_root(cfg.output_root)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    torch.manual_seed(42)

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    print(f"rank={rank}, local_rank={local_rank}, world_size={world_size}, device={device}")

    # 不同 rank 切不同 expert
    expert_indices = list(range(rank, cfg.num_experts, world_size))
    print(f"[Rank {rank}] experts = {expert_indices}")
    print(f"[Rank {rank}] output_root = {cfg.output_root}")

    os.makedirs(cfg.output_root, exist_ok=True)
    process_out_dir = os.path.join(cfg.output_root, f"rank_{rank}")
    os.makedirs(process_out_dir, exist_ok=True)

    print(f"[Rank {rank}] loading act samples from {cfg.act_samples_path} ...")
    act_samples_full = torch.load(cfg.act_samples_path, map_location="cpu")

    overall_summary = []

    with safe_open(cfg.weights_path, framework="pt", device="cpu") as weight_reader:
        for layer_id in range(cfg.layer_start, cfg.layer_end + 1):
            for expert_id in expert_indices:
                try:
                    print(f"\n==== Train layer={layer_id}, expert={expert_id} ====")
                    train_one_expert(
                        layer_id=layer_id,
                        expert_id=expert_id,
                        act_samples_full=act_samples_full,
                        weight_reader=weight_reader,
                        device=device,
                        out_dir=process_out_dir,
                        cfg=cfg,
                        metric=cfg.softsort_metric,
                    )
                    overall_summary.append({
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "status": "ok",
                    })
                    torch.npu.synchronize()
                    torch.npu.empty_cache()

                except Exception:
                    err_msg = traceback.format_exc()
                    print(f"[ERROR] layer={layer_id}, expert={expert_id}")
                    print(err_msg)

                    err_path = os.path.join(
                        process_out_dir,
                        f"layer{layer_id:02d}_expert{expert_id:02d}.err.txt"
                    )
                    with open(err_path, "w", encoding="utf-8") as f:
                        f.write(err_msg)

                    overall_summary.append({
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "status": "error",
                        "err_file": os.path.basename(err_path),
                    })
                    torch.npu.synchronize()
                    torch.npu.empty_cache()

    overall_summary_path = os.path.join(process_out_dir, "rank_summary.json")
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, ensure_ascii=False, indent=2)

    print(f"[Rank {rank}] all done.")


if __name__ == "__main__":
    main() 
