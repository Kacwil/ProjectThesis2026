import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warp as wp

from data_processing_functions import build_ant_pos, build_ant_pos_ordered
from music import music_top_minimal_wp
from classical_doa import process_results, plot

from synth_data import music_cmp_to_ned
from synth_plotting import *
from random_walk import sample_physical_random_walk_NEU, inspect_random_walk

import raytracer as rt
from utils import *
from raytracer.terrain_generation import *
from plotting import *
from antennas import *
from beamformer_warp import *
from utils import load_iq_dataset, get_sampling_order

import time
from tqdm import tqdm

from music import music_top_minimal_wp
from classical_doa import process_results, plot


wp.config.quiet = True
wp.init()
device = "cuda:0" if wp.is_cuda_available() else "cpu"

# Configuration
N_SAMPLES = 50000
max_elevation_deg = 25
num_antennas = 12
carrier_freq = 2402*1e6
rx_origin = np.array([0.0, 0.0, 0.42])
rtconstants = rt.RaytracerConstants(n_rays_mp=int(1e5), n_rays_per_rx=1000, n_rays_per_cell=8, max_bounces=2)

use_cache = False # If false, raytracing will by runned
path_cache = r"temp_RT_OUTS.pkl"

name = f"RT_{int(N_SAMPLES/1000)}k_lowel_temp"
path_save = rf"data/{name}.pkl"

def run_raytracer():

    pos, vel, acc = sample_physical_random_walk_NEU(N_SAMPLES, max_el=max_elevation_deg)

    N = pos[:, 0]
    E = pos[:, 1]
    U = pos[:, 2]

    ned = pos.copy()
    ned[:, 2] *= -1.0

    xy_dist = np.sqrt(N**2 + E**2)
    az = np.arctan2(E, N)
    el = np.arctan2(U, xy_dist)

    ang = np.column_stack([az, el])
    boresight_az = ang[:, 0].mean(axis=0)

    # Generate enviroment
    terrain = UniformTerrain3D.from_xyz(n_waves=1, steps_per_unit=10, device=device, x_size=[-50, 500], y_size=[-10, 10], random=False)
    rx = Receiver(device, origin=rx_origin, orientation=np.array([np.pi/4, 0, boresight_az]))
    #rx.visualize_mesh()
    tx = Transmitter(pos[0, :3], orientation=np.array([0.0, 0.0, 0.0]))
    scene = rt.Scene(tx, rx, terrain, freq=carrier_freq)

    # Main Loop
    IQ_LOS = np.zeros((N_SAMPLES, 12), dtype=np.complex64)
    IQ_MP = np.zeros((N_SAMPLES, 12), dtype=np.complex64)
    DIST = np.zeros(N_SAMPLES, dtype=np.float32)

    print("Raytracing: ...")

    for i in tqdm(range(N_SAMPLES)):

        d_vec = rx_origin - pos[i, :3]
        dist = np.linalg.norm(d_vec)
        d = d_vec / dist

        yaw = np.arctan2(d[1], d[0])
        pitch = -np.arctan2(d[2], np.sqrt(d[0]**2 + d[1]**2))

        new_tx = Transmitter(pos[i, :3], np.array([0.0, pitch, yaw]))
        scene.tx = new_tx

        outs, debugs = rt.raycast(scene, rtconstants)
        outs, debugs = outs.to_numpy(), debugs.to_numpy()

        DIST[i] = dist
        IQ_LOS[i, :] = (outs["rx_v_no_mp"][:, 0] + 1j * outs["rx_v_no_mp"][:, 1]) * dist**2 * 1e6
        IQ_MP[i, :] = (outs["rx_v"][:, 0] + 1j * outs["rx_v"][:, 1]) * dist**2 * 1e6

    IQ_MP_ONLY = IQ_MP - IQ_LOS
    #IQ_combined, mp_gain, final_db = combine_los_mp(IQ_LOS, IQ_MP_ONLY, target_los_to_mp_db=0)

    #IQ_MP = IQ_LOS + 5e6 * (IQ_MP - IQ_LOS)

    out = {
        "IQ_LOS": IQ_LOS,
        "IQ_MP": IQ_MP,
        "NED": ned,
        "ANG": ang,
        "DIST": DIST,
        "boresight_az": boresight_az,
    }

    with open(path_cache, "wb") as f:
        pickle.dump(out, f)

def create_estimates(save=True):
    with open(path_cache, "rb") as f:
        synth = pickle.load(f)


    iq, ant_pos, angles, positions, keys, dt = load_iq_dataset()
    sampling_order = get_sampling_order()
    iq = iq[:, :, 0] + 1j * iq[:, :, 1]

    N = iq.shape[0]
    iq_avg = np.zeros((N, num_antennas), dtype=complex)

    for ant in range(1, num_antennas + 1):
        mask = (sampling_order == ant)
        iq_avg[:, ant - 1] = iq[:, mask].mean(axis=1)

    iq = iq_avg
    phase = np.angle(iq)
    rel_phase = np.angle(np.exp(1j * (phase[:, :, None] - phase[:, None, :])))

    IQ = synth["IQ_MP"]
    PHI = np.angle(IQ)
    REL = np.angle(np.exp(1j * (PHI[:, :, None] - PHI[:, None, :])))


    if True:

        results = []
        T = 8 
        MP = True
        for i in range(T, N_SAMPLES):

            if (i+1) % 100==0:
                print(f"Music: {i}/{N_SAMPLES}")

            if MP:
                X = 1e9 * synth["IQ_MP"][i-T+1:i+1, :]
                #X = mp[i-T+1:i+1, :]
            else:
                X = 1e9 * synth["IQ_LOS"][i-T+1:i+1, :]

            music = music_top_minimal_wp(
                burst_mat=X,
                fr_mhz=2402*1e6,
                az_step=0.5,
                el_step=0.5,
                n_sources=1,
                ant_pos = build_ant_pos()
            )

            music["az_hat_deg"] = music["az_hat_deg"]                   

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

        azhat_ned, elhat_ned = music_cmp_to_ned(
            -azhat,
            elhat,
            synth["boresight_az"],
        )

        inspect_music_hat(
            np.rad2deg(synth["ANG"][T:, 0]),
            azhat_ned,
            np.rad2deg(synth["ANG"][T:, 1]),
            elhat_ned,
        )

        anghat = np.column_stack((
            np.deg2rad(azhat_ned),
            np.deg2rad(elhat_ned),
        )).astype(np.float32)

        iq = np.stack(
            (synth["IQ_MP"].real, synth["IQ_MP"].imag),
            axis=-1,
        ).astype(np.float32)

        out = {
            "IQ": iq[T:],
            "IQ_NO_MP": synth["IQ_LOS"][T:],
            "ANG": synth["ANG"][T:].astype(np.float32),
            "POS": synth["NED"][T:].astype(np.float32),
            "DIST": synth["DIST"][T:],
            "boresight_az": synth["boresight_az"],
            "ANG_HAT": anghat,
        }

        return out

def combine_los_mp(
    iq_los,
    iq_mp,
    target_los_to_mp_db,
):
    """
    Combine LOS and multipath IQ with a desired LOS/MP ratio, since the RT doesnt seem to be weighted properly.
    """

    rms_los = np.sqrt(np.mean(np.abs(iq_los)**2))
    rms_mp  = np.sqrt(np.mean(np.abs(iq_mp)**2))

    mp_gain = rms_los / (
        rms_mp * 10**(target_los_to_mp_db / 20)
    )

    iq_combined = iq_los + mp_gain * iq_mp

    achieved_db = 20 * np.log10(
        np.sqrt(np.mean(np.abs(iq_los)**2)) /
        (
            np.sqrt(np.mean(np.abs(mp_gain * iq_mp)**2))
            + 1e-12
        )
    )

    return iq_combined, mp_gain, achieved_db

if __name__ == "__main__":

    if not use_cache:  
        run_raytracer()

    out = create_estimates(save=True)

    with open(path_save, "wb") as f:
        pickle.dump(out, f)

    print(f"Saved to {path_save}")

    plt.show()