import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from utils import get_sampling_order
from tqdm import tqdm
from utils_coordinates import *


def inspect_music_hat(azgt, azhat, elgt, elhat):
    import numpy as np
    import matplotlib.pyplot as plt

    azgt = np.asarray(azgt)
    azhat = np.asarray(azhat)

    elgt = np.asarray(elgt)
    elhat = np.asarray(elhat)

    # RMSE
    az_rmse = np.sqrt(np.mean((azhat - azgt) ** 2))
    el_rmse = np.sqrt(np.mean((elhat - elgt) ** 2))

    fig = plt.figure(figsize=(10, 5))

    # Azimuth
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(azgt, azgt, label="Az GT")
    ax1.plot(azgt, azhat, label="Az Est")
    ax1.set_xlabel("Azimuth GT [deg]")
    ax1.set_ylabel("Azimuth HAT [deg]")
    ax1.set_title(f"Azimuth\nRMSE = {az_rmse:.2f}°")
    ax1.grid(True)
    ax1.legend()

    # Elevation
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(elgt, elgt, label="El GT")
    ax2.plot(elgt, elhat, label="El Est")
    ax2.set_xlabel("Elevation GT [deg]")
    ax2.set_ylabel("Elevation HAT [deg]")
    ax2.set_title(f"Elevation\nRMSE = {el_rmse:.2f}°")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    print(f"Azimuth RMSE   : {az_rmse:.3f} deg")
    print(f"Elevation RMSE : {el_rmse:.3f} deg")

def inspect_multiple_estimates(azgt, elgt, estimates):
    """
    estimates : dict
        Dictionary of estimators.

        Format:
        {
            "method_name": {
                "az": az_estimates,
                "el": el_estimates,
            },
            ...
        }

    """

    import numpy as np
    import matplotlib.pyplot as plt

    azgt = np.asarray(azgt)
    elgt = np.asarray(elgt)

    fig = plt.figure(figsize=(12, 5))

    # -------------------------
    # Azimuth plot
    # -------------------------
    ax1 = plt.subplot(1, 2, 1)

    # Ideal line
    ax1.plot(azgt, azgt, "k--", linewidth=2, label="Ideal")

    for name, data in estimates.items():
        azhat = np.asarray(data["az"])

        az_rmse = np.sqrt(np.mean((azhat - azgt) ** 2))

        ax1.plot(
            azgt,
            azhat,
            marker="o",
            linestyle="",
            alpha=0.7,
            label=f"{name} (RMSE={az_rmse:.2f}°)"
        )

    ax1.set_xlabel("Azimuth GT [deg]")
    ax1.set_ylabel("Azimuth Estimate [deg]")
    ax1.set_title("Azimuth")
    ax1.grid(True)
    ax1.legend()

    # -------------------------
    # Elevation plot
    # -------------------------
    ax2 = plt.subplot(1, 2, 2)

    ax2.plot(elgt, elgt, "k--", linewidth=2, label="Ideal")

    for name, data in estimates.items():
        elhat = np.asarray(data["el"])

        el_rmse = np.sqrt(np.mean((elhat - elgt) ** 2))

        ax2.plot(
            elgt,
            elhat,
            marker="o",
            linestyle="",
            alpha=0.7,
            label=f"{name} (RMSE={el_rmse:.2f}°)"
        )

    ax2.set_xlabel("Elevation GT [deg]")
    ax2.set_ylabel("Elevation Estimate [deg]")
    ax2.set_title("Elevation")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    # -------------------------
    # Print summary
    # -------------------------
    print("\nRMSE Summary")
    print("-" * 40)

    for name, data in estimates.items():
        azhat = np.asarray(data["az"])
        elhat = np.asarray(data["el"])

        az_rmse = np.sqrt(np.mean((azhat - azgt) ** 2))
        el_rmse = np.sqrt(np.mean((elhat - elgt) ** 2))

        print(
            f"{name:<15} | "
            f"Az RMSE: {az_rmse:7.3f} deg | "
            f"El RMSE: {el_rmse:7.3f} deg"
        )

def inspect_IQ_AMP(real_IQ, synth_IQ, positions, synth_dist, take_mean=True):
    plt.figure(figsize=(7, 5))

    # Convert (..., 2) real/imag format to complex
    if real_IQ.shape[-1] == 2:
        real_IQ_c = real_IQ[..., 0] + 1j * real_IQ[..., 1]
    else:
        real_IQ_c = real_IQ

    if synth_IQ.shape[-1] == 2:
        synth_IQ_c = synth_IQ[..., 0] + 1j * synth_IQ[..., 1]
    else:
        synth_IQ_c = synth_IQ

    # Mean amplitude per burst/sample
    real_amp = np.mean(np.abs(real_IQ_c), axis=1)
    synth_amp = np.mean(np.abs(synth_IQ_c), axis=1)

    # Real distance
    real_dist = np.sqrt(positions[:, 0]**2 + positions[:, 1]**2)

    # Synthetic distance
    if take_mean and np.ndim(synth_dist) > 1:
        synth_x = synth_dist.mean(axis=1)
    else:
        synth_x = synth_dist

    print("real_dist:", real_dist.shape)
    print("real_amp:", real_amp.shape)
    print("synth_x:", np.asarray(synth_x).shape)
    print("synth_amp:", synth_amp.shape)

    plt.scatter(
        real_dist,
        real_amp,
        s=6,
        alpha=0.6,
        label="Real"
    )

    plt.scatter(
        synth_x,
        synth_amp,
        s=6,
        alpha=0.8,
        label="Synthetic"
    )

    plt.xlabel("Distance [m]")
    plt.ylabel("Mean |IQ|")
    plt.title("IQ Amplitude vs Distance")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

def plot_iq_noise_effect(iq_clean, iq_noisy, sample_idx=0):
    """
    iq_clean: (N, M) clean complex IQ
    iq_noisy: (N, M) noisy complex IQ
    sample_idx: which snapshot to visualize
    """
    clean = iq_clean[sample_idx]
    noisy = iq_noisy[sample_idx]
    m = np.arange(clean.size)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    # Complex plane
    axs[0].scatter(clean.real, clean.imag, label="Clean")
    axs[0].scatter(noisy.real, noisy.imag, label="Noisy")
    axs[0].set_xlabel("I")
    axs[0].set_ylabel("Q")
    axs[0].set_title("IQ Complex Plane")
    axs[0].axis("equal")
    axs[0].grid(True)
    axs[0].legend()

    # Amplitude
    axs[1].plot(m, np.abs(clean), marker="o", label="Clean")
    axs[1].plot(m, np.abs(noisy), marker="x", label="Noisy")
    axs[1].set_xlabel("Measurement index")
    axs[1].set_ylabel("Amplitude")
    axs[1].set_title("Amplitude")
    axs[1].grid(True)
    axs[1].legend()

    # Phase
    axs[2].plot(m, np.angle(clean), marker="o", label="Clean")
    axs[2].plot(m, np.angle(noisy), marker="x", label="Noisy")
    axs[2].set_xlabel("Measurement index")
    axs[2].set_ylabel("Phase [rad]")
    axs[2].set_title("Phase")
    axs[2].grid(True)
    axs[2].legend()

    plt.tight_layout()

def plot_iq_amplitude_by_element(iq):
    data = []
    labels = []
    sampling_order = get_sampling_order()

    for elem_id in sorted(np.unique(sampling_order)):
        idx = np.where(sampling_order == elem_id)[0]
        amp = np.abs(iq[:, idx]).ravel()
        amp = amp[np.isfinite(amp)]

        data.append(amp)
        labels.append(elem_id)

    plt.figure(figsize=(10, 4))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.xlabel("Element ID")
    plt.ylabel("|IQ|")
    plt.title("IQ Amplitude by Antenna Element")
    plt.grid(True)
