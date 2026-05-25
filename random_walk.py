import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from tqdm import tqdm
from utils import load_iq_dataset

def sample_physical_random_walk_NEU(
    n_samples,
    acc_std=np.array([np.sqrt(7.5), np.sqrt(5), np.sqrt(7.5)]),   # m/s²
    vel_std_init=np.array([0.0, 0.0, 0.0]),
    vel_limit=np.array([35.0, 1.0, 10.0]),
    acc_limit=np.array([15.0, 15.0, 15.0]),
    inspect=False,
    max_el = 90,
    seed=1111,
):
    rng = np.random.default_rng(seed)
    _ ,_ , angles, positions, _, dt = load_iq_dataset()

    pad_rad = np.deg2rad(0.0)
    pad_meter = 0.0

    az_range = [np.deg2rad(-15), np.deg2rad(15)]
    el_range = [angles[:, 1].min() - pad_rad, min(np.deg2rad(max_el), angles[:, 1].max() + pad_rad)]

    xy_distance_range = [5.0, positions[:, 0].max() + pad_meter]
    height_range = [
        (-positions[:, 1]).min() - pad_meter,
        (-positions[:, 1]).max() + pad_meter,
    ]

    mid_angles = np.array([
        np.mean(az_range),
        np.mean(el_range),
    ])

    mid_dist = np.mean(xy_distance_range)

    mid_position = np.array([
        np.cos(mid_angles[1]) * np.cos(mid_angles[0]) * mid_dist,
        np.cos(mid_angles[1]) * np.sin(mid_angles[0]) * mid_dist,
        np.sin(mid_angles[1]) * mid_dist
    ])

    pos = np.zeros((n_samples, 3), dtype=float)
    vel = np.zeros((n_samples, 3), dtype=float)
    acc = np.zeros((n_samples, 3), dtype=float)

    pos[0] = mid_position
    vel[0] = rng.normal(0.0, vel_std_init, size=3)

    def valid(p):
        N, E, U = p
        xy = np.sqrt(N**2 + E**2)
        az = np.arctan2(E, N)
        el = np.arctan2(U, xy)

        return (
            xy_distance_range[0] <= xy <= xy_distance_range[1]
            and height_range[0] <= U <= height_range[1]
            and az_range[0] <= az <= az_range[1]
            and el_range[0] <= el <= el_range[1]
        )

    print("Random walking...")
    for i in tqdm((range(1, n_samples))):
            
        a = rng.normal(0.0, acc_std, size=3) - 0.1 * vel[i - 1]
        a = np.clip(a, -acc_limit, acc_limit)

        v = 0.95 * vel[i - 1] + a * dt
        v = np.clip(v, -vel_limit, vel_limit)

        p = pos[i - 1] + v * dt + 0.5 * a * dt**2

        if valid(p):
            acc[i] = a
            vel[i] = v
            pos[i] = p

        else:

            acc[i] = 0.5 * (mid_position - pos[i-1]) - 0.25 * pos[i-1] - 0.2 * vel[i - 1]
            acc[i] = np.clip(acc[i], -acc_limit, acc_limit)
            vel[i] = vel[i - 1] + acc[i] * dt
            pos[i] = pos[i - 1] + vel[i] * dt + 0.5 * acc[i] * dt**2


            # bounce/damp if stuck
            #acc[i] = 0.0
            #vel[i] = -0.5 * vel[i - 1] - 0.002*pos[i-1] - 0.002 * (mid_position - pos[i-1])
            #pos[i] = pos[i - 1]

    if inspect:
        inspect_random_walk(pos, xy_distance_range, height_range, az_range, el_range)

    return pos, vel, acc


def inspect_random_walk(pos, xy_distance_range, height_range, az_range, el_range):

    N = pos[:, 0]
    E = pos[:, 1]
    D = pos[:, 2]
    U = -D

    xy_dist = np.sqrt(N**2 + E**2)

    az_min, az_max = az_range
    el_min, el_max = el_range
    h_min, h_max = height_range
    r_min, r_max = xy_distance_range

    fig, axs = plt.subplots(3, 1, figsize=(6.5, 9))

    # ------------------------------------------------------------
    # Plot 1: Top view with azimuth range
    # ------------------------------------------------------------
    axs[0].plot(E, N, linewidth=1, alpha=0.8)
    axs[0].scatter(E[0], N[0], c="green", label="Start", zorder=3)
    axs[0].scatter(E[-1], N[-1], c="red", label="End", zorder=3)

    # Azimuth boundary rays
    ray_len = max(r_max, np.nanmax(xy_dist))

    for az, label in [(az_min, "Az min"), (az_max, "Az max")]:
        e_ray = ray_len * np.sin(az)
        n_ray = ray_len * np.cos(az)

        axs[0].plot(
            [0, e_ray],
            [0, n_ray],
            linestyle="--",
            linewidth=1,
        )

    axs[0].set_xlabel("East [m]")
    axs[0].set_ylabel("North [m]")
    axs[0].set_title("Random Walk Top View")
    axs[0].axis("equal")
    axs[0].grid(True)
    axs[0].legend()

    # ------------------------------------------------------------
    # Plot 2: D coordinate with negative height bounds
    # ------------------------------------------------------------
    axs[1].plot(D, linewidth=1, label="D")

    axs[1].axhline(
        y=h_min,
        linestyle="--",
        linewidth=1,
    )

    axs[1].axhline(
        y=h_max,
        linestyle="--",
        linewidth=1,
    )

    axs[1].set_xlabel("Step")
    axs[1].set_ylabel("Height [m]")
    axs[1].set_title("Height Over Time")
    axs[1].grid(True)

    # ------------------------------------------------------------
    # Plot 3: Distance vs height with elevation range and max height
    # ------------------------------------------------------------
    #axs[2].scatter(xy_dist, D, s=5, alpha=0.6, label="Trajectory")
    axs[2].plot(xy_dist, D)

    x_line = np.linspace(
        max(1e-6, r_min),
        max(r_max, np.nanmax(xy_dist)),
        200
    )

    for el, label in [(el_min, "El min"), (el_max, "El max")]:
        axs[2].plot(
            x_line,
            np.tan(el) * x_line,
            linestyle="--",
            linewidth=1,
            label=label
        )

    axs[2].axhline(
        y=h_max,
        linestyle=":",
        linewidth=1,
    )

    axs[2].set_xlabel("NE Distance [m]")
    axs[2].set_ylabel("Height [m]")
    axs[2].set_title("Distance vs Height")
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig("random_walk.png", format="png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    n_samples = 100000
    pos, vel, acc = sample_physical_random_walk_NEU(n_samples, inspect=True, max_el=20)


    _ ,_ , angles, positions, _, dt = load_iq_dataset()

    print(positions.shape)

    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = np.linalg.norm(vel, axis=1)
    accmag = np.linalg.norm(acc, axis=1)

    print("step mean/p95/max:", step.mean(), np.percentile(step, 95), step.max())
    print("speed mean/p95/max:", speed.mean(), np.percentile(speed, 95), speed.max())
    print("acc mean/p95/max:", accmag.mean(), np.percentile(accmag, 95), accmag.max())


    real_step = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)

    real_speed = real_step / dt
    real_acc = np.diff(real_speed) / dt

    print("REAL step mean/p95/max:",
        real_step.mean(), np.percentile(real_step, 95), real_step.max())

    print("REAL speed mean/p95/max:",
        real_speed.mean(), np.percentile(real_speed, 95), real_speed.max())

    print("REAL acc mean/p95/max:",
        np.mean(np.abs(real_acc)),
        np.percentile(np.abs(real_acc), 95),
        np.max(np.abs(real_acc)))
