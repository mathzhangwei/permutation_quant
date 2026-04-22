import os
import json
import argparse
from typing import Dict, List

import torch
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot per-expert loss curves from saved training results."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root directory that contains rank_* subdirectories or .pt result files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save generated figures and summary. Defaults to <input-root>/plots",
    )
    parser.add_argument(
        "--plot-ema",
        action="store_true",
        help="Also plot loss_ema_history if present.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Figure DPI.",
    )
    return parser.parse_args()


def collect_result_files(input_root: str) -> List[str]:
    result_files = []
    for root, _, files in os.walk(input_root):
        for name in files:
            if name.endswith(".pt") and ".summary." not in name:
                result_files.append(os.path.join(root, name))
    result_files.sort()
    return result_files


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_single_result(
    result: Dict,
    save_path: str,
    plot_ema: bool,
    dpi: int,
) -> Dict:
    key = result["key"]
    layer_id = result["layer_id"]
    expert_id = result["expert_id"]
    loss_history = result.get("loss_history", [])
    loss_ema_history = result.get("loss_ema_history", [])
    baseline = result.get("baseline_identity_loss")
    identity_refined_loss = result.get("identity_refined_loss")
    best_loss = result.get("best_loss_before_refine")
    final_best_loss = result.get("final_best_loss")
    improvement = result.get("improvement_vs_identity")
    best_epoch = result.get("best_epoch")

    if not loss_history:
        raise ValueError(f"{key}: loss_history is empty")

    epochs = list(range(len(loss_history)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, loss_history, label="hard_eval_loss", linewidth=1.8)

    if plot_ema and loss_ema_history:
        ax.plot(epochs, loss_ema_history, label="hard_eval_loss_ema", linewidth=1.5, alpha=0.9)

    if baseline is not None:
        ax.axhline(baseline, color="tab:red", linestyle="--", linewidth=1.5, label="baseline_identity")

    if identity_refined_loss is not None:
        ax.axhline(
            identity_refined_loss,
            color="tab:orange",
            linestyle="--",
            linewidth=1.3,
            label="identity_direct_refine",
        )

    if best_loss is not None:
        ax.axhline(best_loss, color="tab:green", linestyle=":", linewidth=1.3, label="best_before_refine")

    if final_best_loss is not None:
        ax.axhline(final_best_loss, color="tab:purple", linestyle="-.", linewidth=1.3, label="final_best")

    if best_epoch is not None and 0 <= best_epoch < len(loss_history):
        ax.scatter([best_epoch], [loss_history[best_epoch]], color="black", s=24, zorder=5, label="best_epoch")

    ax.set_title(f"Layer {layer_id} Expert {expert_id}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()

    subtitle = f"improve_vs_identity={improvement:.4%}" if improvement is not None else key
    fig.text(0.5, 0.01, subtitle, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)

    return {
        "key": key,
        "layer_id": layer_id,
        "expert_id": expert_id,
        "baseline_identity_loss": baseline,
        "identity_refined_loss": identity_refined_loss,
        "best_loss_before_refine": best_loss,
        "final_best_loss": final_best_loss,
        "improvement_vs_identity": improvement,
        "best_epoch": best_epoch,
        "plot_path": save_path,
    }


def main() -> None:
    args = parse_args()
    input_root = os.path.abspath(args.input_root)
    output_dir = os.path.abspath(args.output_dir or os.path.join(input_root, "plots"))
    ensure_dir(output_dir)

    result_files = collect_result_files(input_root)
    if not result_files:
        raise FileNotFoundError(f"No result .pt files found under: {input_root}")

    summary_rows = []

    for result_path in result_files:
        result = torch.load(result_path, map_location="cpu")
        layer_id = result["layer_id"]
        expert_id = result["expert_id"]

        layer_dir = os.path.join(output_dir, f"layer_{layer_id:02d}")
        ensure_dir(layer_dir)

        save_path = os.path.join(layer_dir, f"layer{layer_id:02d}_expert{expert_id:02d}.png")
        summary_rows.append(
            plot_single_result(
                result=result,
                save_path=save_path,
                plot_ema=args.plot_ema,
                dpi=args.dpi,
            )
        )

    summary_path = os.path.join(output_dir, "plot_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"Found {len(result_files)} result files.")
    print(f"Saved plots to: {output_dir}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
