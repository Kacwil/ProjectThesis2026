import numpy as np
import torch
import time

from .train_walkforward import load_RL_data, get_model_spec
from .train_dataclasses import *
from .train_utils import *

@dataclass(frozen=True)
class EvalResult:
    gt_rad: np.ndarray
    pred_rad: np.ndarray
    gate: np.ndarray | None = None
    expert_deltas: np.ndarray | None = None
    inference_time_s: float | None = None
    inference_time_per_sample_ms: float | None = None

def create_full_eval_dataset(
    model_cfg: ModelConfig,
    data: DataArrays,
):
    spec = get_model_spec(model_cfg)

    data = DataArrays(
        iq=normalize_full(data.iq),
        angles_hat=data.angles_hat,
        positions=normalize_full(data.positions),
        angles=data.angles,
    )

    model = spec.build_model(model_cfg)
    dataset = spec.build_dataset(data, False, model_cfg, False)

    return model, dataset

def load_checkpoint(model, checkpoint_path: str, device: str) -> None:
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

def run_eval_inference(model, loader, device: str) -> EvalResult:
    gt = []
    pred = []
    gates = []
    deltas = []

    model.eval()

    # Synchronize before timing if using CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    with torch.no_grad():
        for batch in loader:
            targets = batch["targets"].cpu().numpy()

            out = model(batch["features"])

            gt.append(targets)
            pred.append(out["ang_hat"].detach().cpu().numpy())

            if "gate" in out:
                gates.append(out["gate"].detach().cpu().numpy())

            if "deltas" in out:
                expert_deltas = torch.stack(out["deltas"], dim=1)
                deltas.append(expert_deltas.detach().cpu().numpy())

    # Synchronize after timing if using CUDA
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    inference_time_s = time.perf_counter() - t0

    gt_arr = np.concatenate(gt, axis=0)
    pred_arr = np.concatenate(pred, axis=0)
    deltas_arr = np.concatenate(deltas, axis=0) if deltas else None
    gate = np.concatenate(gates, axis=0) if gates else None

    inference_time_per_sample_ms = 1000.0 * inference_time_s / len(gt_arr)

    print(f"Inference time total      : {inference_time_s:.4f} s")
    print(f"Inference time per sample : {inference_time_per_sample_ms:.4f} ms")

    return EvalResult(
        gt_rad=gt_arr,
        pred_rad=pred_arr,
        gate=gate,
        expert_deltas=deltas_arr,
        inference_time_s=inference_time_s,
        inference_time_per_sample_ms=inference_time_per_sample_ms,
    )

def plot_relative_expert_contribution_vs_elevation(result: EvalResult) -> None:
    if result.expert_deltas is None or result.gate is None:
        return

    result = sort_eval_by_elevation(result)

    elevation_deg = np.rad2deg(result.gt_rad[:, 1])
    deltas = result.expert_deltas          # (N, E, 2)
    gates = result.gate                    # (N, E)

    if gates.ndim == 1:
        gates = gates[:, None]

    gated_deltas = deltas * gates[:, :, None]       # (N, E, 2)
    contribution_mag = np.linalg.norm(gated_deltas, axis=2)  # (N, E)

    denom = contribution_mag.sum(axis=1, keepdims=True)
    ratio = contribution_mag / (denom + 1e-8)

    plt.figure(figsize=(12, 4))

    for i in range(ratio.shape[1]):
        plt.scatter(
            elevation_deg,
            ratio[:, i],
            s=8,
            alpha=0.6,
            label=f"Expert {i}",
        )

    plt.xlabel("Ground truth elevation (deg)")
    plt.ylabel("Fraction of total gated expert contribution")
    plt.title("Relative expert contribution vs elevation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def sort_eval_by_elevation(result: EvalResult) -> EvalResult:
    idx = np.argsort(result.gt_rad[:, 1], kind="stable")

    return EvalResult(
        gt_rad=result.gt_rad[idx],
        pred_rad=result.pred_rad[idx],
        gate=None if result.gate is None else result.gate[idx],
        expert_deltas=(
            None if result.expert_deltas is None
            else result.expert_deltas[idx]
        ),
        inference_time_s=result.inference_time_s,
        inference_time_per_sample_ms=result.inference_time_per_sample_ms,
    )

def plot_post_training_result(result: EvalResult) -> None:
    result = sort_eval_by_elevation(result)

    gt_deg = np.rad2deg(result.gt_rad)
    pred_deg = np.rad2deg(result.pred_rad)

    if result.gate is not None:
        gate = np.asarray(result.gate)

        if gate.ndim == 1:
            gate = gate[:, None]

        if gate.ndim > 2:
            gate = gate.reshape(gate.shape[0], -1)

        plt.figure(figsize=(7, 4))

        for j in range(gate.shape[1]):
            label = "Gate" if gate.shape[1] == 1 else f"Gate {j}"
            plt.scatter(
                gt_deg[:, 1],
                gate[:, j],
                s=8,
                alpha=0.6,
                label=label,
            )

        plt.xlabel("Ground truth elevation (deg)")
        plt.ylabel("Gate value")
        plt.title("Elevation vs Gate")
        plt.grid(True, alpha=0.3)

        if gate.shape[1] > 1:
            plt.legend()

    plt.figure(figsize=(5, 5))
    plt.scatter(gt_deg[:, 1], pred_deg[:, 1], s=8, alpha=0.6)

    mn = min(gt_deg[:, 1].min(), pred_deg[:, 1].min())
    mx = max(gt_deg[:, 1].max(), pred_deg[:, 1].max())
    plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)

    plt.xlabel("Ground truth elevation (deg)")
    plt.ylabel("Predicted elevation (deg)")
    plt.title("Elevation: Prediction vs Ground Truth")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    plt.show()

def post_training_evaluation(
    device: str,
    roots: dict,
    split_cfg: SplitConfig,
    model_cfg: ModelConfig,
    checkpoint_path: str = r"Trainer\model_cache.pth",
) -> EvalResult:
    data = load_RL_data(roots, split_cfg)
    model, dataset = create_full_eval_dataset(model_cfg, data)

    load_checkpoint(model, checkpoint_path, device)

    loader = make_loader(
        dataset,
        batch_size=model_cfg.batch_size,
    )

    result = run_eval_inference(model, loader, device)
    plot_post_training_result(result)
    plot_relative_expert_contribution_vs_elevation(result)

    return result
