import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import copy
from numpy.typing import NDArray
import numpy as np


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def make_loader(dataset, batch_size=64, shuffle=False, seed=1111):
    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def criterion(out, target):

    pred_loss = F.huber_loss(out["ang_hat"], target)

    # Punish low-confidence predictions:
    #gate_loss = out["gate"].abs().mean()
    #raw_delta_loss = out["expert_deltas"].abs().mean()

    return pred_loss #+ 0.01 * gate_loss


def train_one_epoch(model, loader, optimizer, device="cpu"):
    model.train()

    total_loss = 0.0
    n_samples = 0

    for batch in loader:

        out = model(batch["features"])
        loss = criterion(out, batch["targets"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        B = out["ang_hat"].shape[0]
        total_loss += loss.item() * B
        n_samples += B

    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()

    total_loss = 0.0
    n_samples = 0

    gt = []
    pred = []

    for batch in loader:

        out = model(batch["features"])
        loss = criterion(out, batch["targets"])

        B = out["ang_hat"].shape[0]
        total_loss += loss.item() * B
        n_samples += B

        gt.append(batch["targets"].cpu().numpy())
        pred.append(out["ang_hat"].cpu().numpy())

    gt = np.concatenate(gt, axis=0)
    pred = np.concatenate(pred, axis=0)

    return total_loss / n_samples, gt, pred


def train_model(
    model,
    train_loader,
    val_loader=None,
    test_loader=None,
    epochs=50,
    lr=1e-3,
    device="cpu",
):
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "test_loss": [],
        "best_val_loss": 1.0e9,
        "best_weights": None,
    }

    model.to(device)

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss, _, _ = evaluate(
                model=model,
                loader =val_loader,
                device=device,
            )
            if val_loss < history["best_val_loss"]:
                history["best_val_loss"] = val_loss
                history["best_weights"] = copy.deepcopy(model.state_dict())



        if test_loader is not None:

            if history["best_weights"] is not None:
                model.load_state_dict(history["best_weights"])

            test_loss, _, _ = evaluate(
                model=model,
                loader=test_loader,
                device=device,
            )
            history["test_loss"].append(test_loss)

        log_msg = f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f}"

        if val_loader is not None:
            log_msg += f" | val_loss={val_loss:.6f}"

        if test_loader is not None:
            log_msg += f" | test_loss={test_loss:.6f}"

        print(log_msg)

    torch.save(model.state_dict(), r"Trainer\model_cache.pth")
    return history

def angles_rad_to_deg(gt_rad, pred_rad):
    gt_deg = np.rad2deg(gt_rad)
    pred_deg = np.rad2deg(pred_rad)

    gtaz = gt_deg[:, 0]
    gtel = gt_deg[:, 1]
    predaz = pred_deg[:, 0]
    predel = pred_deg[:, 1]

    return gtaz, gtel, predaz, predel


def compute_angle_rmse_deg(gtaz, gtel, predaz, predel, elevation_bias_deg=0.0):
    predel_corr = predel + elevation_bias_deg

    az_err = wrap180(predaz - gtaz)
    el_err = predel_corr - gtel

    rmse_az = np.sqrt(np.mean(az_err ** 2))
    rmse_el = np.sqrt(np.mean(el_err ** 2))

    return rmse_az, rmse_el

def plot_angle_timeseries(t, gtaz, gtel, predaz, predel, elevation_bias_deg=0.0):
    predel_corr = predel + elevation_bias_deg

    rmse_az, rmse_el = compute_angle_rmse_deg(
        gtaz=gtaz,
        gtel=gtel,
        predaz=predaz,
        predel=predel,
        elevation_bias_deg=elevation_bias_deg,
    )

    print(f"RMSE Azimuth   : {rmse_az:.3f} deg")
    print(f"RMSE Elevation : {rmse_el:.3f} deg")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    axes[0].plot(t, gtel, "k-", linewidth=1.2, label="Ground truth")
    axes[0].scatter(t, predel_corr, s=8, alpha=0.8, label="Predicted")
    axes[0].set_ylabel("Elevation angle (deg)")
    axes[0].set_title("Elevation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, gtaz, "k-", linewidth=1.2, label="Ground truth")
    axes[1].scatter(t, predaz, s=8, alpha=0.8, label="Predicted")
    axes[1].set_ylabel("Azimuth angle (deg)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_title("Azimuth")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    return rmse_az, rmse_el


def wrap180(x):
    return (x + 180.0) % 360.0 - 180.0


def rolling_mean(x, window=50):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(x[lo:i+1])
    return out

def rolling_rmse(x, window=50):
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)

    for i in range(len(x)):
        lo = max(0, i - window + 1)
        out[i] = np.sqrt(np.mean(x[lo:i+1] ** 2))

    return out

def compute_music_errors():
    df = pd.read_csv(r"C:\Uni\V26\Prosjekt\Simulator\20220607\music.csv")

    az_hat = df["az_hat_deg"].to_numpy()
    el_hat = df["el_hat_deg"].to_numpy()
    az_gt  = df["az_gt_deg"].to_numpy()
    el_gt  = df["el_gt_deg"].to_numpy()

    az_err = wrap180(az_hat - az_gt)
    el_err = el_hat - el_gt

    return az_err, el_err, az_hat, el_hat

def plot_walkforward_oos(
    t,
    gtaz,
    gtel,
    predaz,
    predel,
    fold_ids=None,
    elevation_bias_deg=0.0,
    error_window=50,
):
    predel_corr = predel + elevation_bias_deg

    az_err = wrap180(predaz - gtaz)
    el_err = predel_corr - gtel

    abs_az_err = np.abs(az_err)
    abs_el_err = np.abs(el_err)

    rmse_az = np.sqrt(np.mean(az_err ** 2))
    rmse_el = np.sqrt(np.mean(el_err ** 2))

    music_az_err, music_el_err, music_az_hat, music_el_hat = compute_music_errors()

    model_color = "tab:blue"
    music_color = "tab:orange"

    print(f"Walk-forward OOS RMSE Azimuth   : {rmse_az:.3f} deg")
    print(f"Walk-forward OOS RMSE Elevation : {rmse_el:.3f} deg")

    # ------------------------------------------------------------
    # 4 subplots now
    # ------------------------------------------------------------
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # ------------------------------------------------------------
    # Elevation
    # ------------------------------------------------------------
    axes[0].plot(t, gtel, "k-", linewidth=1.2, label="Ground truth")
    axes[0].scatter(t, predel_corr, s=8, alpha=0.8, label="Predicted (OOS)")
    axes[0].scatter(t, music_el_hat[t], s=8, alpha=0.25,color=music_color,label="MUSIC")
    axes[0].set_ylabel("Elevation (deg)")
    axes[0].set_title(f"Elevation (RMSE={rmse_el:.2f} deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 50)

    # ------------------------------------------------------------
    # Azimuth
    # ------------------------------------------------------------
    axes[1].plot(t, gtaz, "k-", linewidth=1.2, label="Ground truth")
    axes[1].scatter(t, predaz, s=8, alpha=0.8, label="Predicted (OOS)")
    axes[1].scatter(t, music_az_hat[t], s=8, alpha=0.25,color=music_color,label="MUSIC")
    axes[1].set_ylabel("Azimuth (deg)")
    axes[1].set_title(f"Azimuth (RMSE={rmse_az:.2f} deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_ylim(0, -10)

    # ------------------------------------------------------------
    # Azimuth error progression
    # ------------------------------------------------------------

    az_curve = rolling_rmse(az_err, window=error_window)
    music_az_curve = rolling_rmse(music_az_err[t], error_window)

    axes[2].plot(t, az_curve, linewidth=1.995, label=f"Model rolling ({error_window})")
    axes[2].plot(t, music_az_curve, linewidth=1.995, label=f"Music rolling ({error_window})")
    axes[2].axhline(music_az_curve.mean(), linestyle="--", linewidth=1.5, color=music_color, label="Music AVG")
    axes[2].axhline(az_curve.mean(), linestyle="--", linewidth=1.5, color=model_color, label="Model AVG")

    axes[2].set_ylabel("Az error (deg)")
    axes[2].set_title("Azimuth error progression")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # ------------------------------------------------------------
    # Elevation error progression
    # ------------------------------------------------------------
    el_curve = rolling_rmse(el_err, window=error_window)
    music_el_curve = rolling_rmse(music_el_err[t], error_window)

    axes[3].plot(t, el_curve, linewidth=2.0, label=f"Model rolling ({error_window})")
    axes[3].plot(t, music_el_curve, linewidth=1.995, label=f"Music rolling ({error_window})")
    axes[3].axhline(music_el_curve.mean(), linestyle="--", linewidth=1.5, color=music_color, label="Music AVG")
    axes[3].axhline(el_curve.mean(), linestyle="--", linewidth=1.5, color=model_color, label="Model AVG")

    axes[3].set_ylabel("El error (deg)")
    axes[3].set_xlabel("Sample index")
    axes[3].set_title("Elevation error progression")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    # ------------------------------------------------------------
    # Fold boundaries
    # ------------------------------------------------------------
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids)
        boundaries = np.where(np.diff(fold_ids) != 0)[0]
        for b in boundaries:
            x = t[b + 1]
            for ax in axes:
                ax.axvline(x=x, linewidth=1.0, alpha=0.6)

    plt.tight_layout()
    plt.show()

    return {
        "rmse_az": rmse_az,
        "rmse_el": rmse_el,
        "az_err": az_err,
        "el_err": el_err,
    }

def load_iq_dataset(pkl_path):
    import pickle

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    ant_pos = raw["ANT_POS_ORDERED"].T
    data = raw["DATA"]

    keys = sorted(data.keys())
    n = len(keys)

    iq = np.zeros((n, 82, 2), dtype=np.float32)
    angles = np.zeros((n, 2), dtype=np.float32)
    positions = np.zeros((n, 2), dtype=np.float32)

    for i, burst_id in enumerate(keys):
        burst = data[burst_id]
        x = burst["MATCHED"]["X"]

        iq[i, :, 0] = np.real(x)
        iq[i, :, 1] = np.imag(x)

        angles[i, 0] = burst["NAV"]["azimuth"]
        angles[i, 1] = burst["NAV"]["elevation"]

        positions[i, 0] = burst["NAV"]["dist_NE"]
        positions[i, 1] = burst["NAV"]["D"]

    return iq, ant_pos.astype(np.float32), angles, positions, keys


def normalize_full(data):
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True) + 1e-6
    return (data - mean) / std


def normalize_split(data, indices):

    train = data[indices.train]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True) + 1e-6

    return (
        (data[indices.train] - mean) / std,
        (data[indices.val] - mean) / std,
        (data[indices.test] - mean) / std,
    )

def get_baseline_data(root):
    baseline = pd.read_csv(root)

    az_hat_deg = baseline["az_hat_deg"].to_numpy()
    el_hat_deg = baseline["el_hat_deg"].to_numpy() + 1.7

    angles_hat = np.column_stack([np.deg2rad(az_hat_deg),np.deg2rad(el_hat_deg),]).astype(np.float32)
    return angles_hat

