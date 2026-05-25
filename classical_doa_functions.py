import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from beamformer_warp import *
from music import *
import warp as wp
from data_processing_functions import build_ant_pos, get_sampling_order
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
from synth_plotting import inspect_music_hat




# --- Functions for creating data snapshots ---
def burst_to_snapshots(data, burst_id, sampling_order, ref_ant=11, antennas=None, average_ref_ant=False):
    burst = np.asarray(data[burst_id]["MATCHED"]["X"])
    sampling_order = np.asarray(sampling_order)

    grouped = defaultdict(list)

    for ant, value in zip(sampling_order, burst):
        grouped[int(ant)].append(value)

    grouped = {
        ant: np.asarray(values)
        for ant, values in grouped.items()
    }

    if antennas is None:
        antennas = sorted(grouped.keys())

    if average_ref_ant:
        length_ants = [a for a in antennas if a != ref_ant]
    else:
        length_ants = antennas

    T_local = min(len(grouped[a]) for a in length_ants)

    snapshots = np.zeros((T_local, len(antennas)), dtype=burst.dtype)

    if average_ref_ant:
        ref_value = np.mean(grouped[ref_ant])

    for j, ant in enumerate(antennas):
        if ant == ref_ant and average_ref_ant:
            snapshots[:, j] = ref_value
        else:
            snapshots[:, j] = grouped[ant][:T_local]

    return snapshots

def make_T_snapshots(
    data,
    burst_id,
    sampling_order,
    T=8,
    ref_ant=11,
    antennas=None,
):
    all_snapshots = []

    keys = sorted(data.keys())

    try:
        idx = keys.index(burst_id)
    except ValueError:
        raise ValueError(f"burst_id={burst_id} not found")

    while len(all_snapshots) < T:

        if idx < 0:
            raise ValueError(
                f"Not enough bursts available to make {T} snapshots."
            )

        current_id = keys[idx]

        local_snapshots = burst_to_snapshots(
            data=data,
            burst_id=current_id,
            sampling_order=sampling_order,
            ref_ant=ref_ant,
            antennas=antennas,
        )

        for row in reversed(local_snapshots):
            all_snapshots.append(row)

            if len(all_snapshots) == T:
                break

        idx -= 1

    return np.asarray(all_snapshots)

# --- Plotting functions ---
def inspect(path):

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}, try running the DOA generation functions."

    df = pd.read_csv(path)
    print(df)

    t = df["t_sec"]
    azgt = df["az_gt_deg"]
    elgt = df["el_gt_deg"]

    azhat = df["az_hat_deg"]
    elhat = df["el_hat_deg"]

    plot_music_predictions(t, azgt, elgt, azhat, elhat)
    inspect_music_hat(azgt, azhat, elgt, elhat)
    plt.show()

def plot(t, azgt, elgt, azhat, elhat, data=None, scatter_gt=False):
    elevation_bias_deg = 1.7

    az_err = wrap180(azhat - azgt)
    el_err = (elhat + elevation_bias_deg) - elgt

    rmse_az = np.sqrt(np.mean(az_err ** 2))
    rmse_el = np.sqrt(np.mean(el_err ** 2))

    print(f"RMSE Elevation : {rmse_el:.3f} deg")
    print(f"RMSE Azimuth   : {rmse_az:.3f} deg")

    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    if not scatter_gt:
        axes[0].plot(t, elgt, color="black", linewidth=1.5, alpha=1, label="Ground truth")
    else:
        axes[0].scatter(t, elgt, s=8, alpha=0.8, label="Ground truth")

    axes[0].scatter(t,elhat + elevation_bias_deg,s=8,alpha=0.8,label="Predicted")

    axes[0].set_ylabel("Elevation angle (deg)")
    axes[0].set_title("Elevation")
    axes[0].set_ylim(-10, 50)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if not scatter_gt:
        axes[1].plot(t, azgt, color="black",linewidth=1.5, alpha=1, label="Ground truth")
    else:
        axes[1].scatter(t, azgt ,s=8, alpha=0.8, label="Ground truth")

    axes[1].scatter(t, azhat, s=8, alpha=0.8, label="Predicted")

    axes[1].set_ylabel("Azimuth angle (deg)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Azimuth")
    axes[1].set_ylim(0, -10)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    height = []
    dist = []

    if data is not None:

        for key, val in data.items():
            height.append(-val["NAV"]["D"])
            dist.append(val["NAV"]["dist_NE"])

        fig, ax = plt.subplots()

        ax.plot(dist, height, "k-", linewidth=1.2)
        ax.set_ylabel("Height above receiver (m)")
        ax.set_xlabel("NE distance from receiver (m)")
        ax.set_title("Flight Path")
        ax.grid(True, alpha=0.3)
        ax.legend()


        plt.tight_layout()
        plt.show()

def plot_smoothed(t, azgt, elgt, azhat, elhat, window=5):

    el_smooth = (elhat.rolling(window=window, center=True, min_periods=1).mean())
    az_smooth = (azhat.rolling(window=window, center=True, min_periods=1).mean())
    plot(t, azgt, elgt, az_smooth, el_smooth)

def plot_music_predictions(t,gtaz,gtel,music_azhat,music_elhat,elevation_bias_deg=0.0):
    """
    Plot MUSIC azimuth/elevation predictions against ground truth.

    music_azhat and music_elhat can be either:
      - scalars
      - arrays/lists with same indexing as t
    """

    t = np.asarray(t)
    gtaz = np.asarray(gtaz)
    gtel = np.asarray(gtel)

    def _as_prediction_array(pred, t):
        if np.isscalar(pred):
            return np.full_like(t, pred, dtype=float)

        pred = np.asarray(pred)

        # If full-length array indexed by sample index, select using t
        if pred.ndim == 1 and pred.shape[0] > len(t):
            return pred[t]

        # If already aligned with t
        if pred.ndim == 1 and pred.shape[0] == len(t):
            return pred

        raise ValueError(
            "Prediction must be either a scalar, an array aligned with t, "
            "or a full-length array indexable by t."
        )

    music_az = _as_prediction_array(music_azhat, t)
    music_el = _as_prediction_array(music_elhat, t) + elevation_bias_deg

    az_err = wrap180(music_az - gtaz)
    el_err = music_el - gtel

    rmse_az = np.sqrt(np.mean(az_err ** 2))
    rmse_el = np.sqrt(np.mean(el_err ** 2))

    fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)

    # ------------------------------------------------------------
    # Elevation
    # ------------------------------------------------------------
    axes[0].plot(t, gtel, "k-", linewidth=1.2, label="Ground truth")
    axes[0].scatter(t, music_el, s=8, alpha=0.7, label="Prediction")
    axes[0].set_ylabel("Elevation (deg)")
    axes[0].set_title(f"Elevation prediction (RMSE={rmse_el:.2f} deg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_ylim(0, 50)

    # ------------------------------------------------------------
    # Azimuth
    # ------------------------------------------------------------
    axes[1].plot(t, gtaz, "k-", linewidth=1.2, label="Ground truth")
    axes[1].scatter(t, music_az, s=8, alpha=0.7, label="Prediction")
    axes[1].set_ylabel("Azimuth (deg)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_title(f"Azimuth prediction (RMSE={rmse_az:.2f} deg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def plot_optimization_result(rmse_df, label="T"):

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    az_idx = rmse_df["rmse_az_deg"].idxmin()
    el_idx = rmse_df["rmse_el_deg"].idxmin()

    az_T_opt = rmse_df.loc[az_idx, label]
    az_rmse_opt = rmse_df.loc[az_idx, "rmse_az_deg"]

    el_T_opt = rmse_df.loc[el_idx, label]
    el_rmse_opt = rmse_df.loc[el_idx, "rmse_el_deg"]

    axes[0].plot(rmse_df[label], rmse_df["rmse_az_deg"], marker="o")
    axes[0].scatter(az_T_opt, az_rmse_opt, s=80, zorder=5)
    axes[0].axvline(az_T_opt, linestyle="--", alpha=0.7)
    axes[0].set_ylabel("Az RMSE [deg]")
    axes[0].set_title(f"Azimuth RMSE vs {label} | optimum {label}={az_T_opt}, RMSE={az_rmse_opt:.3f}°")
    axes[0].grid(True)

    axes[1].plot(rmse_df[label], rmse_df["rmse_el_deg"], marker="o")
    axes[1].scatter(el_T_opt, el_rmse_opt, s=80, zorder=5)
    axes[1].axvline(el_T_opt, linestyle="--", alpha=0.7)
    axes[1].set_xlabel(label)
    axes[1].set_ylabel("El RMSE [deg]")
    axes[1].set_title(f"Elevation RMSE vs {label} | optimum {label}={el_T_opt}, RMSE={el_rmse_opt:.3f}°")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"Azimuth optimum:   {label}={az_T_opt}, RMSE={az_rmse_opt:.4f} deg")
    print(f"Elevation optimum: {label}={el_T_opt}, RMSE={el_rmse_opt:.4f} deg")
