import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from beamformer_warp import *
from music import *
import warp as wp
from data_processing_functions import build_ant_pos, get_sampling_order
from tqdm import tqdm
import time

from classical_doa_functions import *

"""
Script to create beamformer/MUSIC estimates, from experimental data.
"""

data_path = r"data\experimental_data.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

wp.config.quiet = True
wp.init()
device = "cuda:0" if wp.is_cuda_available() else "cpu"


def process_results(results):
    "Transform the predictions from antenna frame into NED frame. Return as pandas dataframe."

    rows = []

    for r in results:
        nav = r["NAV"]
        ts = getattr(nav, "name", pd.NaT)

        rows.append({
            "burst_id": r["burst_id"],
            "time": ts,
            "az_hat_raw_deg": float(r["az_hat_raw_deg"]),
            "el_hat_raw_deg": float(r["el_hat_raw_deg"]),
            "az_gt_deg": float(r["az_gt_deg"]),
            "el_gt_deg": float(r["el_gt_deg"]),
        })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.sort_values("burst_id").reset_index(drop=True)

    if df["time"].notna().any():
        t0 = df["time"].dropna().iloc[0]
        df["t_sec"] = (df["time"] - t0).dt.total_seconds()
    else:
        df["t_sec"] = np.arange(len(df), dtype=float)

    df["az_hat_deg"] = df["el_hat_raw_deg"] * np.sin(np.deg2rad(df["az_hat_raw_deg"]))
    df["el_hat_deg"] = df["el_hat_raw_deg"] * np.cos(np.deg2rad(df["az_hat_raw_deg"]))

    return df

def beamformer_estimator(data):

    """
    Function for running Beamformer
    """

    results = []
    for i, burst_id in enumerate(tqdm(data.keys(), desc="Processing bursts")):

        azgt_deg = np.rad2deg(data[burst_id]["NAV"]["azimuth"])
        elgt_deg = np.rad2deg(data[burst_id]["NAV"]["elevation"])

        bf = beamform_top_minimal_wp(
            burst=data[burst_id],
            ant_pos_ordered=ant_pos_ordered,
            fr_mhz=2402,
            device="cuda",
            az_step=1.0,
            el_step=1.0,
            refine=True
        )

        results.append({
            "burst_id": burst_id,
            "az_hat_raw_deg": bf["az_hat_deg"],
            "el_hat_raw_deg": bf["el_hat_deg"],
            "az_gt_deg": azgt_deg,
            "el_gt_deg": elgt_deg,
            "NAV": data[burst_id]["NAV"],
        })
    df = process_results(results)
    df.to_csv(r"20220607\bf.csv")
    return df

def music_estimator(data, ant_pos_ordered, mode=0 ,T=8, fr_base_mhz=2402, az_step=0.5, el_step=0.5, n_sources=1, save=True):
    """
    Function for running MUSIC
    """
    results = []
    keys = sorted(data.keys())
    burst_mat = np.zeros((T, 82))
    sampling_order = get_sampling_order()

    for i, burst_id in enumerate(tqdm(data.keys(), desc="Processing bursts")):

        if i < T - 1:
            continue

        if mode == 0: # Cut each burst into 6 snapshots.

            snapshots = make_T_snapshots(
                data=data,
                burst_id=burst_id,
                sampling_order=sampling_order,
                T=T,
                ref_ant=11,
            )

            burst_mat = snapshots

        elif mode == 1: # Work with full burst (T,82)

            window_keys = keys[i - T + 1 : i + 1]
            burst_mat = np.stack([np.asarray(data[k]["MATCHED"]["X"], dtype=np.complex64) for k in window_keys],axis=0,)

        else: # Average the burst per antenna into (T,12)
            window_keys = keys[i - T + 1 : i + 1]
            burst_mat = np.stack([np.asarray(data[k]["MATCHED"]["X_AVG"], dtype=np.complex64) for k in window_keys],axis=0,)
            

        azgt_deg = np.rad2deg(data[burst_id]["NAV"]["azimuth"])
        elgt_deg = np.rad2deg(data[burst_id]["NAV"]["elevation"])

        music = music_top_minimal_wp(
            burst_mat=burst_mat,
            ant_pos_ordered=ant_pos_ordered,
            fr_mhz=fr_base_mhz * 1e6 + data[burst_id]["CFO"],
            az_step=az_step,
            el_step=el_step,
            n_sources=n_sources,
            use_ref_ant=True,
            ant_pos = build_ant_pos(),
        )

        results.append({
            "burst_id": burst_id,
            "az_hat_raw_deg": music["az_hat_deg"],
            "el_hat_raw_deg": music["el_hat_deg"],
            "az_gt_deg": azgt_deg,
            "el_gt_deg": elgt_deg,
            "NAV": data[burst_id]["NAV"],
        })

    df = process_results(results)
    if save:
        df.to_csv(r"20220607\music.csv")
    return df

def optimize_music_snapshot_length(
    data,
    ant_pos_ordered,
    modes=(0, 1, 2),
    T_values=range(1, 60, 2),
    n_sources=1,
    elevation_bias_deg=1.7,
    out_dir="music_T_optimization",
):
    """
    Optimize MUSIC snapshot matrix length T for different burst aggregation modes.

    For each mode and snapshot count T, MUSIC is evaluated against ground-truth
    azimuth and elevation angles. Azimuth, elevation, combined angular RMSE,
    and execution time are stored.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for mode in modes:
        for T in T_values:
            start_time = time.perf_counter()

            df = music_estimator(
                data,
                ant_pos_ordered,
                mode=mode,
                T=T,
                n_sources=n_sources,
                save=False,
            )

            runtime_s = time.perf_counter() - start_time

            az_gt = df["az_gt_deg"].to_numpy(dtype=float)
            el_gt = df["el_gt_deg"].to_numpy(dtype=float)

            az_hat = df["az_hat_deg"].to_numpy(dtype=float)
            el_hat = df["el_hat_deg"].to_numpy(dtype=float)

            az_err = wrap180(az_hat - az_gt)
            el_err = (el_hat + elevation_bias_deg) - el_gt

            rmse_az = np.sqrt(np.mean(az_err**2))
            rmse_el = np.sqrt(np.mean(el_err**2))
            rmse_total = np.sqrt(rmse_az**2 + rmse_el**2)

            rows.append({
                "mode": mode,
                "T": T,
                "rmse_az_deg": rmse_az,
                "rmse_el_deg": rmse_el,
                "rmse_total_deg": rmse_total,
                "runtime_s": runtime_s,
                "n_estimates": len(df),
            })

            print(
                f"mode={mode}, T={T:2d} | "
                f"RMSE az={rmse_az:.3f} deg, "
                f"el={rmse_el:.3f} deg, "
                f"total={rmse_total:.3f} deg | "
                f"time={runtime_s:.2f} s"
            )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / "music_T_optimization_all_results.csv", index=False)

    opt_df = extract_music_optima(results_df)
    opt_df.to_csv(out_dir / "music_T_optimization_optima.csv", index=False)

    plot_music_T_optimization(results_df, opt_df, out_dir)

    return results_df, opt_df


def extract_music_optima(results_df):
    """
    Extract optimal T values for azimuth RMSE, elevation RMSE, total RMSE,
    and runtime for each mode.
    """

    opt_rows = []

    for mode, mode_df in results_df.groupby("mode"):
        az_row = mode_df.loc[mode_df["rmse_az_deg"].idxmin()]
        el_row = mode_df.loc[mode_df["rmse_el_deg"].idxmin()]
        total_row = mode_df.loc[mode_df["rmse_total_deg"].idxmin()]
        time_row = mode_df.loc[mode_df["runtime_s"].idxmin()]

        opt_rows.append({
            "mode": mode,

            "az_T_opt": az_row["T"],
            "az_rmse_opt_deg": az_row["rmse_az_deg"],

            "el_T_opt": el_row["T"],
            "el_rmse_opt_deg": el_row["rmse_el_deg"],

            "total_T_opt": total_row["T"],
            "total_rmse_opt_deg": total_row["rmse_total_deg"],

            "fastest_T": time_row["T"],
            "fastest_runtime_s": time_row["runtime_s"],
        })

    return pd.DataFrame(opt_rows)


def plot_music_T_optimization(results_df, opt_df, out_dir):
    """
    Plot RMSE and computation time as functions of snapshot count T.
    """

    out_dir = Path(out_dir)

    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    metric_info = [
        ("rmse_az_deg", "Azimuth RMSE [deg]", "Azimuth RMSE vs Snapshot Count $T$"),
        ("rmse_el_deg", "Elevation RMSE [deg]", "Elevation RMSE vs Snapshot Count $T$"),
        ("rmse_total_deg", "Total RMSE [deg]", "Total RMSE vs Snapshot Count $T$"),
        ("runtime_s", "Runtime [s]", "Computation Time vs Snapshot Count $T$"),
    ]

    mode_labels = {
        0: "Mini-burst",
        1: "Full burst",
        2: "Averaged burst",
    }

    for ax, (metric, ylabel, title) in zip(axes, metric_info):
        for mode, mode_df in results_df.groupby("mode"):
            ax.plot(
                mode_df["T"],
                mode_df[metric],
                marker="o",
                label=mode_labels.get(mode, f"Mode {mode}"),
            )

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Snapshot count $T$")

    plt.tight_layout()

    fig.savefig(out_dir / "music_T_optimization_modes.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "music_T_optimization_modes.png", dpi=300, bbox_inches="tight")

    plt.show()



if __name__ == "__main__":

    run_bf = False
    inspect_bf = False

    run_MUSIC = False
    inspect_MUSIC = False

    optimize_t = False
    optimize_n = False

    save_result = False

    t = 8 # Number of Music snapshots
    n = 1 # Number of Music sources

    # 0 = cut a burst into 6 snapshots (12), 1 = use full burst matrix (82), 2 = average burst per antenna (12)
    mode = 2 # Use mode 2 by default.

    ant_pos_ordered = data["ANT_POS_ORDERED"]
    data = data["DATA"]

    print(len(data), "samples are loaded from experimental data")

    if run_bf:
        t0 = time.perf_counter()
        df = beamformer_estimator(data)
        print("Beamformer complete, time to compute:", time.perf_counter()-t0)
        if save_result:
            df.to_csv(r"data\bf.csv")

    if run_MUSIC:
        t0 = time.perf_counter()
        df = music_estimator(data, ant_pos_ordered, save=True, T=t, n_sources=n, mode=2)
        print("Music complete, time to compute:", time.perf_counter()-t0)
        if save_result:
            df.to_csv(r"data\music.csv")

    if inspect_bf:
        print("\n Beamformer Results:")
        inspect(r"data\bf.csv")

    if inspect_MUSIC:
        print("\n Music Results:")
        inspect(r"data\music.csv")

    if optimize_t:
        rmse_rows = []
        for t in range(1, 20, 1):
            df = music_estimator(data, ant_pos_ordered, mode=2, T=t, n_sources=1, save=False)

            azgt = df["az_gt_deg"].to_numpy()
            elgt = df["el_gt_deg"].to_numpy()

            azhat = df["az_hat_deg"].to_numpy()
            elhat = df["el_hat_deg"].to_numpy()

            elevation_bias_deg = 0

            az_err = wrap180(azhat - azgt)
            el_err = (elhat + elevation_bias_deg) - elgt

            rmse_az = np.sqrt(np.mean(az_err ** 2))
            rmse_el = np.sqrt(np.mean(el_err ** 2))

            rmse_rows.append({
                "T": n,
                "rmse_az_deg": rmse_az,
                "rmse_el_deg": rmse_el,
            })
        rmse_df = pd.DataFrame(rmse_rows)
        plot_optimization_result(rmse_df)

    if optimize_n:
        rmse_rows = []
        for t in range(1, 5, 1):
            df = music_estimator(data, ant_pos_ordered, mode=2, T=t, n_sources=1, save=False)

            azgt = df["az_gt_deg"].to_numpy()
            elgt = df["el_gt_deg"].to_numpy()

            azhat = df["az_hat_deg"].to_numpy()
            elhat = df["el_hat_deg"].to_numpy()

            elevation_bias_deg = 0

            az_err = wrap180(azhat - azgt)
            el_err = (elhat + elevation_bias_deg) - elgt

            rmse_az = np.sqrt(np.mean(az_err ** 2))
            rmse_el = np.sqrt(np.mean(el_err ** 2))

            rmse_rows.append({
                "T": n,
                "rmse_az_deg": rmse_az,
                "rmse_el_deg": rmse_el,
            })
        rmse_df = pd.DataFrame(rmse_rows)
        plot_optimization_result(rmse_df)

    #experimental mode comparison:
    if True:

        results_df, opt_df = optimize_music_snapshot_length(
            data=data,
            ant_pos_ordered=ant_pos_ordered,
            modes=(0, 1, 2),
            T_values=range(1, 50, 2),
            n_sources=1,
            elevation_bias_deg=1.7,
            out_dir="music_T_optimization",
        )

        print(opt_df)


