from scipy.io import loadmat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing_functions import *
import pickle
from pathlib import Path

from classical_doa_functions import *


"""
Script to read, process, clean and save the experimental data.
"""

root_uav = rf"C:\Uni\V26\Prosjekt\Simulator\data\20220607\sentilog_uav\20220607\112059\combined\matlab_01\sensor_0.mat"
root_rx = rf"C:\Uni\V26\Prosjekt\Simulator\data\20220607\sentilog_base\20220607\112115\combined\matlab_01\sensor_3.mat"
root_save = rf"data\experimental_data.pkl"

folder = Path("data")
folder.mkdir(parents=True, exist_ok=True)

def main():

    pd.set_option("display.float_format", "{:.10f}".format)

    # --- NAV ---
    uav = loadmat(root_uav, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    relative_nav = pd.DataFrame(uav["ublox"]["NAV_RELPOSNED"])
    nav = process_nav(relative_nav)

    # --- RX array ---
    rx = loadmat(root_rx, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    rx = prepare_rx(rx)

    cfos, _,  _ = estimate_cfo_per_burst(rx)
    rx["f_cfo"] = rx["burst_id"].map(cfos).astype(np.float64)

    # --- Final Prep ---
    bursts = prepare_bursts(nav, rx)
    with open(root_save, "wb") as f:
        pickle.dump(bursts, f)

    print(f"Saved to {root_save}")

def plot():

    with open(root_save, "rb") as f:
        data = pickle.load(f)

    data = data["DATA"]

    N, E, D = [], [], []
    az, el = [], []
    I, Q = [], []
    A = []

    for _, burst in data.items():

        az.append(burst["NAV"]["azimuth"])
        el.append(burst["NAV"]["elevation"])

        N.append(burst["NAV"]["N"])
        E.append(burst["NAV"]["E"])
        D.append(burst["NAV"]["D"])

        I.append(np.real(burst["MATCHED"]["X"]))
        Q.append(np.imag(burst["MATCHED"]["X"]))

        A.append(
            (
                np.sqrt(
                    np.real(burst["MATCHED"]["X"][0:9])**2
                    + np.imag(burst["MATCHED"]["X"][0:9])**2
                )
            ).mean()
        )

    N = np.asarray(N)
    E = np.asarray(E)
    D = np.asarray(D)

    U = -D
    step = np.arange(len(U))

    NE_distance = np.sqrt(N**2 + E**2)

    fig, axs = plt.subplots(3, 1, figsize=(6.5, 9))

    # ------------------------------------------------------------
    # East-North
    # ------------------------------------------------------------
    axs[0].plot(E, N, linewidth=1)

    axs[0].scatter(
        E[0],
        N[0],
        c="green",
        label="Start",
        zorder=3
    )

    axs[0].scatter(
        E[-1],
        N[-1],
        c="red",
        label="End",
        zorder=3
    )

    # Zoom around E variation
    e_pad = 0.5
    axs[0].set_xlim(
        np.min(E) - e_pad,
        np.max(E) + e_pad
    )

    axs[0].set_xlabel("East [m]")
    axs[0].set_ylabel("North [m]")
    axs[0].set_title("East-North Trajectory")
    axs[0].grid(True)
    axs[0].legend()

    # ------------------------------------------------------------
    # Step-Height
    # ------------------------------------------------------------
    axs[1].plot(step, U, linewidth=1)

    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Height [m]")
    axs[1].set_title("Height vs Step")
    axs[1].grid(True)

    # ------------------------------------------------------------
    # NE Distance - Height
    # ------------------------------------------------------------
    axs[2].scatter(
        NE_distance,
        U,
        s=5,
        alpha=0.6
    )

    axs[2].set_xlabel("NE Distance [m]")
    axs[2].set_ylabel("Height [m]")
    axs[2].set_title("NE Distance vs Height")
    axs[2].grid(True)

    plt.tight_layout()

    plt.savefig(
        "trajectory_statistics.png",
        format="png",
        bbox_inches="tight"
    )

    print(f"Azimuth   min/max [deg]: "
        f"{np.rad2deg(np.min(az)):.2f} / "
        f"{np.rad2deg(np.max(az)):.2f}")

    print(f"Elevation min/max [deg]: "
        f"{np.rad2deg(np.min(el)):.2f} / "
        f"{np.rad2deg(np.max(el)):.2f}")

    print(f"Height    min/max [m]: "
        f"{np.min(U):.2f} / "
        f"{np.max(U):.2f}")

    print(f"NE dist   min/max [m]: "
        f"{np.min(NE_distance):.2f} / "
        f"{np.max(NE_distance):.2f}")
    
    plt.show()

def inspect_nav():

    with open(root_save, "rb") as f:
        data = pickle.load(f)

    data = data["DATA"]

    N, E, D, T = [], [], [], []

    for burst_id in sorted(data.keys()):

        N.append(data[burst_id]["NAV"]["N"])
        E.append(data[burst_id]["NAV"]["E"])
        D.append(data[burst_id]["NAV"]["D"])
        T.append(data[burst_id]["BURST_TS"])

    df = pd.DataFrame({
        "ts": pd.to_datetime(T),
        "N": N,
        "E": E,
        "D": D
    }).sort_values("ts").reset_index(drop=True)

    # Variable timestep [s]
    dt = df["ts"].diff().dt.total_seconds()



    windows = range(1, 51, 1)

    vel_results = []
    acc_results = []

    vel_var_results = []
    acc_var_results = []
    acc_std_results = []

    selected_window = 10

    pos_smooth = df[["N", "E", "D"]].rolling(
        selected_window,
        center=True
    ).mean()

    # Velocity using variable dt
    vel = pos_smooth.diff().div(dt, axis=0)

    # Acceleration using variable dt
    dt_vel = dt.copy()
    acc = vel.diff().div(dt_vel, axis=0)

    print(vel.abs().max())
    print(acc.abs().max())

    for window in windows:

        pos_smooth = df[["N", "E", "D"]].rolling(
            window,
            center=True
        ).mean()

        vel = pos_smooth.diff().div(dt, axis=0)

        acc = vel.diff().div(dt, axis=0)

        vel_results.append(vel)
        acc_results.append(acc)

        vel_var_results.append(vel.var())
        acc_var_results.append(acc.var())
        acc_std_results.append(acc.std())

    vel_var_df = pd.DataFrame(vel_var_results, index=windows)
    acc_var_df = pd.DataFrame(acc_var_results, index=windows)
    acc_std_df = pd.DataFrame(acc_std_results, index=windows)

    # Find windows corresponding to min, median, max acceleration std
    selected_windows = [
        windows[0],                    # min
        selected_window,
        windows[-1]                    # max
    ]

    fig, axs = plt.subplots(3, 1, figsize=(6.5, 9))

    labels = ["Min Window","Proposed", "Max Window"]
    pos = df[["N", "E", "D"]]

    # Plot N trajectories
    for window, label in zip(selected_windows, labels):

        pos_smooth = pos.rolling(window, center=True).mean()

        axs[0].plot(
            pos_smooth["N"],
            label=f"{label} ({window})"
        )

    axs[0].set_title("North Position Smoothing")
    axs[0].set_xlabel("Sample")
    axs[0].set_ylabel("N")
    axs[0].legend()
    axs[0].grid(True)

    # Plot D trajectories
    for window, label in zip(selected_windows, labels):

        pos_smooth = pos.rolling(window, center=True).mean()

        axs[1].plot(
            pos_smooth["D"],
            label=f"{label} ({window})"
        )

    axs[1].set_title("Down Position Smoothing")
    axs[1].set_xlabel("Sample")
    axs[1].set_ylabel("D")
    axs[1].legend()
    axs[1].grid(True)

    # Acceleration std
    axs[2].plot(windows, acc_var_df["N"], label="N")
    axs[2].plot(windows, acc_var_df["E"], label="E")
    axs[2].plot(windows, acc_var_df["D"], label="D")
    axs[2].set_title("Acceleration Variance vs Window Size")
    axs[2].set_xlabel("Rolling Window")
    axs[2].set_ylabel("Variance")
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    #plot()
    #inspect_nav()