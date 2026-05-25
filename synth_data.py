import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from data_processing_functions import build_ant_pos, build_ant_pos_ordered
from music import music_top_minimal_wp
from classical_doa import process_results
from tqdm import tqdm
from utils_coordinates import *
from random_walk import sample_physical_random_walk_NEU
from utils import load_iq_dataset
from synth_plotting import *

C = 299_792_458.0
FC = 2402 * 1e6
N_SAMPLES = 200000
RX_HEIGHT = 0.5
MAX_EL_DEG = 90

PATH = r"data\experimental_data.pkl"
MP = True
SAVE = None
SAVE = r"C:\Uni\V26\Prosjekt\Simulator\20220607\synth_500k_mp.pkl"

SAMPLING = np.array([
    11,11,11,11,11,11,11,11,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8,7,6,5,
    12,1,2,10,3,9,4,8
], dtype=int)

def violated_bounds(p, mid, xy_distance_range, height_range, az_range, el_range):

    N, E, U = p

    xy = np.sqrt(N**2 + E**2)
    az = np.arctan2(E, N)
    el = np.arctan2(U, xy)

    barrier_acceleration = np.zeros(3)
    direction_to_mid = mid - p
    direction_to_mid /= np.linalg.norm(direction_to_mid) + 1e-9

    if xy < xy_distance_range[0]:
        barrier_acceleration += direction_to_mid

    if xy > xy_distance_range[1]:
        barrier_acceleration += direction_to_mid

    if U < height_range[0]:
        barrier_acceleration[2] += 1.0

    if U > height_range[1]:
        barrier_acceleration[2] -= 1.0

    if az < az_range[0]:
        barrier_acceleration[1] += 1.0

    if az > az_range[1]:
        barrier_acceleration[0] -= 1.0

    if el < el_range[0]:
        barrier_acceleration[2] += 1.0

    if el > el_range[1]:
        barrier_acceleration[2] -= 1.0

    return barrier_acceleration

def sample_start_pos(az_range, el_range, xy_distance_range, height_range, max_tries=1000):
    for _ in range(max_tries):
        az = np.random.uniform(*az_range)
        el = np.random.uniform(*el_range)
        r_xy = np.random.uniform(*xy_distance_range)

        # convert to NEU
        N = np.cos(az) * r_xy
        E = np.sin(az) * r_xy
        U = np.tan(el) * r_xy

        # check height constraint
        if height_range[0] <= U <= height_range[1]:
            return np.array([N, E, U])

    raise RuntimeError("Failed to sample valid start position")

def get_world_antenna_pos(boresight_az, inspect=False):
    ant_pos = build_ant_pos() # 3, 12

    # Antenna local z-axis points along boresight
    z_ant_world = np.array([
        np.cos(boresight_az),   # North
        np.sin(boresight_az),   # East
        0.0                     # Up
    ])

    # Choose local x-axis as world Up
    x_ant_world = np.array([0.0, 0.0, 1.0])

    # Complete right-handed basis: x × y = z
    y_ant_world = np.cross(x_ant_world, z_ant_world)
    y_ant_world /= np.linalg.norm(y_ant_world)

    # World-from-antenna-local rotation
    R_world_ant = np.column_stack([
        x_ant_world,
        y_ant_world,
        z_ant_world
    ])

    # Convert antenna element offsets into world NEU frame
    ant_offsets_world = R_world_ant @ ant_pos   # (3, 12)
    ant_offsets_world[2, :] += 0.42

    N_ant = ant_offsets_world[0, :]
    E_ant = ant_offsets_world[1, :]

    if inspect:
        plt.figure()

        # antenna element positions
        plt.scatter(E_ant, N_ant, label="Antenna elements")

        # label indices (optional, useful for debugging geometry)
        for i in range(len(E_ant)):
            plt.text(E_ant[i], N_ant[i], str(i))

        # boresight direction (unit vector in NE plane)
        N_bore = np.cos(boresight_az)
        E_bore = np.sin(boresight_az)

        # draw arrow from origin
        plt.quiver(
            0, 0,
            E_bore, N_bore,
            angles='xy',
            scale_units='xy',
            scale=1,
            label="Boresight"
        )

        # formatting
        plt.xlabel("East")
        plt.ylabel("North")
        plt.title("Antenna Geometry (XY) with Boresight")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()


    return ant_offsets_world

def get_direction_vectors(pos, ant_pos_world):

    vec = pos[:, None, :] - ant_pos_world.T[None, :, :]
    dist = np.linalg.norm(vec, axis=2)
    unit_vec = vec / dist[:, :, None]

    return vec, unit_vec, dist

def get_IQ(dist, ref_idx=10):
    """
    dist: (N, M) distances from each antenna element
    fc: carrier frequency [Hz]
    ref_idx: reference antenna index

    returns:
        iq: (N, M) complex steering vectors
    """
    wavelength = C / FC
    k = 2 * np.pi / wavelength
    delta_dist = dist - dist[:, [ref_idx]]
    iq = np.exp(-1j * k * delta_dist)
    return iq

def scale_amplitude(IQ, dist):
    amp = 250.0 * np.exp(-0.025 * dist) + 25.0 # Simple amplitude distance model
    #amp = amp * np.random.lognormal(mean=0.0, sigma=0.0, size=dist.shape)
    return amp * IQ

def expand_sampling_order(iq_12, snr_db=None):
    """
    iq_12: (N, 12) complex IQ
    snr_db: desired SNR in dB. If None, no noise.

    returns:
        iq_82: (N, 82)
    """
    idx = SAMPLING - 1
    iq_82 = iq_12[:, idx]

    if snr_db is not None:
        signal_power = np.mean(np.abs(iq_82)**2, axis=1, keepdims=True)
        snr_linear = 10 ** (snr_db / 10)

        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*iq_82.shape) +
            1j * np.random.randn(*iq_82.shape)
        )

        iq_82 = iq_82 + noise

    return iq_82

def LOS_IQ(dist):

    phi0 = np.random.uniform(0.0, 2*np.pi, size=(dist.shape[0], 1))
    return scale_amplitude(get_IQ(dist) * np.exp(1j * phi0), dist)
    
def TWO_PATH_IQ(dist, dist_reflected):
    wavelength = C / FC
    k = 2 * np.pi / wavelength

    gamma = 0.4 * np.exp(1j * np.pi)

    los = scale_amplitude(np.exp(-1j * k * dist), dist)
    reflected = gamma * scale_amplitude(np.exp(-1j * k * dist_reflected), dist_reflected)

    return los + reflected

def neu_to_music_angles(pos_neu):
    """
    pos_neu: (..., 3), [N, E, U]

    returns:
        az_music: radians
        el_music: radians

    MUSIC convention:
        v = [sin(el)*cos(az), sin(el)*sin(az), cos(el)]
    """
    pos_neu = np.asarray(pos_neu)

    N = pos_neu[..., 0]
    E = pos_neu[..., 1]
    U = pos_neu[..., 2]

    r = np.linalg.norm(pos_neu, axis=-1)

    az_music = np.arctan2(E, N)
    el_music = np.arccos(U / np.maximum(r, 1e-12))

    return az_music, el_music

def synthetic_neu_to_music_comparison_angles(pos_neu, boresight_az, origin_neu=np.array([0.0, 0.0, 0.42])):
    """
    Convert synthetic NEU position to the same postprocessed angle convention
    used by results_to_dataframe().

    pos_neu: (N, 3), world NEU target positions
    boresight_az: antenna boresight azimuth in world NE plane [rad]
    origin_neu: antenna origin in world NEU

    returns:
        az_cmp_deg: (N,)
        el_cmp_deg: (N,)
        az_raw_deg: (N,)
        el_raw_deg: (N,)
    """
    R_world_ant = R_world_from_ant(boresight_az)

    # Direction from antenna origin to target in world NEU
    d_world = pos_neu - origin_neu[None, :]
    d_world = d_world / np.linalg.norm(d_world, axis=1, keepdims=True)

    # Convert world direction into antenna-local frame
    d_local = d_world @ R_world_ant

    x = d_local[:, 0]
    y = d_local[:, 1]
    z = d_local[:, 2]

    az_raw = np.arctan2(y, x)
    el_raw = np.arccos(np.clip(z, -1.0, 1.0))

    az_raw_deg = np.rad2deg(az_raw)
    el_raw_deg = np.rad2deg(el_raw)

    # Same transform as results_to_dataframe()
    az_cmp_deg = el_raw_deg * np.sin(np.deg2rad(az_raw_deg))
    el_cmp_deg = el_raw_deg * np.cos(np.deg2rad(az_raw_deg))

    return az_cmp_deg, el_cmp_deg, az_raw_deg, el_raw_deg

def music_cmp_to_ned(az_cmp_deg, el_cmp_deg, boresight_az):
    """
    Convert MUSIC comparison-frame angles back to world NED az/el.

    NED convention:
        position/vector = [N, E, D]
        az = atan2(E, N)
        el = atan2(-D, horizontal distance)

    Inputs:
        az_cmp_deg, el_cmp_deg:
            MUSIC/postprocessed comparison-frame components in degrees.

        boresight_az:
            antenna boresight azimuth in world NE plane [rad].

    Returns:
        az_ned_deg:
            azimuth in world NED frame, degrees, atan2(E, N)

        el_ned_deg:
            elevation angle, degrees, positive upward,
            atan2(-D, sqrt(N^2 + E^2))
    """

    az_cmp_deg = np.asarray(az_cmp_deg)
    el_cmp_deg = np.asarray(el_cmp_deg)

    # Inverse of:
    # az_cmp = el_raw * sin(az_raw)
    # el_cmp = el_raw * cos(az_raw)
    el_raw_deg = np.sqrt(az_cmp_deg**2 + el_cmp_deg**2)
    az_raw_deg = np.rad2deg(np.arctan2(az_cmp_deg, el_cmp_deg))

    az_raw = np.deg2rad(az_raw_deg)
    el_raw = np.deg2rad(el_raw_deg)

    # MUSIC local spherical convention:
    # el_raw is polar angle from local +z, not elevation from horizon.
    x_local = np.sin(el_raw) * np.cos(az_raw)
    y_local = np.sin(el_raw) * np.sin(az_raw)
    z_local = np.cos(el_raw)

    d_local = np.column_stack([x_local, y_local, z_local])

    R_world_ant = R_ned_from_ant(boresight_az)

    # If d_local = d_world @ R_world_ant,
    # then d_world = d_local @ R_world_ant.T
    d_world = d_local @ R_world_ant.T

    N = d_world[:, 0]
    E = d_world[:, 1]
    D = d_world[:, 2]

    az_ned = np.arctan2(E, N)
    el_ned = np.arctan2(-D, np.sqrt(N**2 + E**2))

    return np.rad2deg(az_ned), np.rad2deg(el_ned)

def make_music_snapshots(dist_i, reflected_dist_i=None, T=1, snr_db=200):
    """
    dist_i: (12,)
    reflected_dist_i: (12,) or None

    returns:
        burst_mat: (T, 12)
    """
    rows = []

    for _ in range(T):
        if reflected_dist_i is None:
            x = LOS_IQ(dist_i[None, :])
        else:
            x = TWO_PATH_IQ(dist_i[None, :], reflected_dist_i[None, :])

        if snr_db is not None:
            signal_power = np.mean(np.abs(x)**2, axis=1, keepdims=True)
            noise_power = signal_power / (10 ** (snr_db / 10))

            noise = np.sqrt(noise_power / 2) * (
                np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
            )

            x = x + noise

        rows.append(x[0])

    return np.stack(rows, axis=0)

def main(inspect=True):

    iq, ant_pos, angles, positions, keys, _ = load_iq_dataset()

    pos_neu, _ ,_ = sample_physical_random_walk_NEU(N_SAMPLES, inspect=True, max_el=MAX_EL_DEG)
    pos_ned = neu_to_ned(pos_neu)

    N = pos_ned[:, 0]
    E = pos_ned[:, 1]
    U = pos_ned[:, 2]

    xy_dist = np.sqrt(N**2 + E**2)

    az, el = az_el_from_ned(pos_ned)

    ang = np.column_stack([az, el])
    boresight_az = az.mean()

    ant_pos_ned = get_NED_antenna_pos(boresight_az, inspect=True)

    vec, unit_vec, dist = get_direction_vectors(pos_ned, ant_pos_ned) # (100,12,3) (100,12,3) (100,12)

    reflected_pos = pos_ned.copy()
    reflected_pos[:, 2] = -reflected_pos[:, 2]
    reflected_vec, reflected_unit_vec, reflected_dist = get_direction_vectors(reflected_pos, ant_pos_ned) # (100,12,3) (100,12,3) (100,12)

    IQ = TWO_PATH_IQ(dist, reflected_dist)

    IQ_C = expand_sampling_order(IQ)
    IQ_N = expand_sampling_order(IQ, snr_db=10)

    iq = iq[:, :, 0] + 1j * iq[:, :, 1]

    LOS_IQ12 = LOS_IQ(dist)
    MP_IQ12 = TWO_PATH_IQ(dist, reflected_dist)

    results = []
    temp = {}
    prev = 0.0
    idx=0
    T = 8
    for i in tqdm((range(T, N_SAMPLES))):

        if MP:
            X = MP_IQ12[i-T+1:i+1, :]
        else:
            X = LOS_IQ12[i-T+1:i+1, :]

        music = music_top_minimal_wp(
            burst_mat=X,
            fr_mhz=2402*1e6,
            az_step=0.5,
            el_step=0.5,
            n_sources=1,
            ant_pos = build_ant_pos()
        )

        temp[i] = {
            "NAV": {
                "D": -pos_ned[i, 2],
                "dist_NE": np.sqrt(pos_ned[i, 0]**2 + pos_ned[i, 1]**2)

        }}                       

        results.append({
            "burst_id": i,
            "az_hat_raw_deg": music["az_hat_deg"],
            "el_hat_raw_deg": music["el_hat_deg"],
            "az_gt_deg": 0.0,
            "el_gt_deg": 0.0,

            "NAV": None,
        })

    df = process_results(results)

    azhat = df["az_hat_deg"]
    elhat = df["el_hat_deg"]

    IQ_complex = MP_IQ12.astype(np.complex64)

    IQ = np.stack(
        [IQ_complex.real, IQ_complex.imag],
        axis=-1
    ).astype(np.float32)   # (N, 12, 2)

    azhat_ned, elhat_ned = music_cmp_to_ned(azhat, elhat, boresight_az)

    out = {
        "IQ": IQ[T:, :],          # (N, 12, 2)
        "ANG": ang[T:, :].astype(np.float32),               # (N, 2) → [az, el] in radians (NEU)
        "POS": pos_ned[T:, :].astype(np.float32),               # (N, 3) → [N, E, U]
        "ANG_HAT": np.column_stack([np.deg2rad(azhat_ned), np.deg2rad(elhat_ned)]).astype(np.float32),  # (N, 2) degrees (postprocessed)
    }

    print(MP_IQ12.shape)
    print(ang.shape)
    print(pos_ned.shape)
    print(np.column_stack([azhat, elhat]).astype(np.float32).shape)

    if SAVE is not None:
        save(out)
        print(f"Saved to {SAVE}")

    if inspect:
        inspect_IQ_AMP(iq, MP_IQ12, positions, dist)
        #inspect_music_hat(az_gt_cmp_deg[T:], azhat,el_gt_cmp_deg[T:], elhat)
        inspect_music_hat(np.rad2deg(ang[T:, 0]), azhat_ned, np.rad2deg(ang[T:, 1]), elhat_ned)
        #inspect_random_walk(pos_ned)
        plt.show()

def save(out):
    with open(SAVE, "wb") as f:
        pickle.dump(out, f)

if __name__ == "__main__":
    main()