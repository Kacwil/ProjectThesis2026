import numpy as np
import warp as wp
import pandas as pd
import matplotlib.pyplot as plt
import time
import raytracer as rt

from utils import *
from raytracer.terrain_generation import *
from plotting import *
from antennas import *
from beamformer_warp import *

wp.config.quiet = True
wp.init()
device = "cuda:0" if wp.is_cuda_available() else "cpu"

# --- Raytracer Configuration ---
rtconstants = rt.RaytracerConstants(n_rays_mp=1, n_rays_per_rx=1000, max_bounces=5)

# --- Scene Configuration ---
terrain = UniformTerrain3D.from_xyz(n_waves=1, steps_per_unit=10, device=device, x_size=[-50, 250], y_size=[-10, 10])
rx = Receiver(device, origin=np.array([0.0, 0.0, 0.42]), orientation=np.array([0, np.deg2rad(10), 0]))
rx.visualize_mesh()
tx = Transmitter(origin=np.array([0, 0, 0]), orientation=np.array([0, 0, np.pi]))
scene = rt.Scene(tx, rx, terrain)


# --- Main loop ---
if False:

    t0 = time.perf_counter()
    outs, debugs = raycast(scene, rtconstants)
    t1 = time.perf_counter()

    print("Raycast time:", t1 - t0)

    outs = outs.to_numpy()
    debugs = debugs.to_numpy()

    hits = debugs["end_hits"]
    unique, counts = np.unique(hits, return_counts=True)
    print(np.asarray((unique, counts)).T)

    terrain_hitcount = debugs["terrain_hitcounts"]
    plot_terrain_heatmap(terrain_hitcount)

if False:

    outs, debugs = raycast(scene, rtconstants)
    outs, debugs = outs.to_numpy(), debugs.to_numpy()

    rx_hit = debugs["end_hits"]
    path_lengths = debugs["path_lengths"]

    valid = (rx_hit != 14) & (path_lengths > 0)
    valid_lengths = path_lengths[valid]

    print(path_lengths)

    xmin = valid_lengths.min()
    xmax = valid_lengths.max()
    bins = np.linspace(xmin, xmax, 100)

    plt.figure(figsize=(10,6))

    for rx_id in range(1, 13):
        mask = (rx_hit == rx_id)
        sub_lengths = path_lengths[mask]

        if sub_lengths.size > 0:
            plt.hist(
                sub_lengths,
                bins=bins,
                density=True,
                linewidth=1.5,
                label=f"RX {rx_id}"
            )

    plt.xlabel("Path Length")
    plt.ylabel("Count")
    plt.title("Path Length Distribution (All Subreceivers)")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

if False:

    for i in [10000000]:
        for j in [1000]:

            rtconstants.n_rays_mp = i
            rtconstants.n_rays_per_rx = j

            outs, debugs = raycast(scene, rtconstants)
            outs, debugs = outs.to_numpy(), debugs.to_numpy() 

            voltages_los = outs["rx_v_no_mp"]
            voltages_mp = outs["rx_v"] - outs["rx_v_no_mp"]

            amps_los = np.sqrt(np.abs(voltages_los[:, 0])**2 + np.abs(voltages_los[:, 1])**2)
            amps_mp  = np.sqrt(np.abs(voltages_mp[:, 0])**2 + np.abs(voltages_mp[:, 1])**2)

            print(outs["rx_v"])
            print(outs["rx_v_no_mp"])

            if False:
                print("Number of MP rays:", i)
                print("Number of rx rays per rx (12 rx exist)", j)
                print("Mean Amplitude LOS:", amps_los.mean())
                print("Mean Amplitude MP:", amps_mp.mean())

    amps = np.vstack([amps_mp, amps_los]).T
    amps = amps / amps.max().max()
    plt.figure(figsize=(6, 5))
    plt.imshow(amps, aspect='auto')
    plt.colorbar(label='|a|')

    plt.xlabel("Path index")
    plt.ylabel("RX element")
    plt.title("Amplitude heatmap")
    plt.tight_layout()
    plt.show()

if False:
    az_fixed_deg = -180.0
    az = np.deg2rad(np.array([az_fixed_deg], dtype=np.float64))
    #az = np.deg2rad(np.arange(-180.0, 180.0, 0.25, dtype=np.float32))
    el = np.deg2rad(np.arange(-90.0, 90.0 + 0.25, 0.25))

    outs, debugs = raycast(scene, rtconstants)
    outs = outs.to_numpy()

    # Received scalar voltages
    voltages_combined = outs["rx_v"]
    v_combined = voltages_combined[:, 0] + 1j * voltages_combined[:, 1]

    voltages_los = outs["rx_v_no_mp"]
    v_los = voltages_los[:, 0] + 1j * voltages_los[:, 1]

    voltages_mp = voltages_combined - voltages_los
    v_mp = voltages_mp[:, 0] + 1j * voltages_mp[:, 1]

    # Beam spectra

    t0 = time.perf_counter()
    P_combined = beamform_power(v_combined, az, el, rx.origin, rx.sub_centers)[:, 0]
    P_los = beamform_power(v_los, az, el, rx.origin, rx.sub_centers)[:, 0]
    P_mp = beamform_power(v_mp, az, el, rx.origin, rx.sub_centers)[:, 0]
    t1 = time.perf_counter()

    print("3x Beamforming time:", t1 - t0)

    # Peak estimates
    el_hat_combined = el[int(np.argmax(P_combined))]
    el_hat_los = el[int(np.argmax(P_los))]
    el_hat_mp = el[int(np.argmax(P_mp))]

    # True elevation
    tx_pos = scene.tx.origin.astype(np.float64)
    k = scene.rx.origin.astype(np.float64) - tx_pos
    k /= np.linalg.norm(k)
    true_el = np.arctan2(k[2], np.sqrt(k[0] ** 2 + k[1] ** 2))

    # Received energies
    E_combined = np.sum(np.abs(v_combined) ** 2)
    E_los = np.sum(np.abs(v_los) ** 2)
    E_mp = np.sum(np.abs(v_mp) ** 2)

    # Confidence metrics
    def confidence_metrics(P: np.ndarray):
        Pmax = np.max(P)
        Pmean = np.mean(P)
        Psorted = np.sort(P)
        P2 = Psorted[-2] if len(Psorted) > 1 else Pmax
        conf_mean = Pmax / (Pmean + 1e-30)
        conf_second = Pmax / (P2 + 1e-30)
        return conf_mean, conf_second

    conf_combined = confidence_metrics(P_combined)
    conf_los = confidence_metrics(P_los)
    conf_mp = confidence_metrics(P_mp)

    # Plot spectra
    plt.figure(figsize=(10, 6))
    plt.plot(np.rad2deg(el), P_combined, label="Combined")
    plt.plot(np.rad2deg(el), P_los, label="LOS only")
    plt.plot(np.rad2deg(el), P_mp, label="Multipath only")

    plt.axvline(np.rad2deg(true_el), color="k", linestyle="--", label="True el")
    plt.axvline(np.rad2deg(el_hat_combined), color="r", linestyle="--", alpha=0.8, label="Combined peak")
    plt.axvline(np.rad2deg(el_hat_los), color="g", linestyle="--", alpha=0.8, label="LOS peak")
    plt.axvline(np.rad2deg(el_hat_mp), color="b", linestyle="--", alpha=0.8, label="MP peak")

    plt.xlabel("Elevation (deg)")
    plt.ylabel("Power")
    plt.title("Beamformer Elevation Spectra")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print diagnostics
    print(f"True elevation: {np.rad2deg(true_el):.3f} deg")
    print()

    print("=== Peak elevations ===")
    print(f"Combined:      {np.rad2deg(el_hat_combined):.3f} deg")
    print(f"LOS only:      {np.rad2deg(el_hat_los):.3f} deg")
    print(f"Multipath only:{np.rad2deg(el_hat_mp):.3f} deg")
    print()

    print("=== Total received energy sum(|v|^2) ===")
    print(f"Combined:      {E_combined:.6e}")
    print(f"LOS only:      {E_los:.6e}")
    print(f"Multipath only:{E_mp:.6e}")
    print(f"MP / LOS:      {E_mp / (E_los + 1e-30):.6e}")
    print()

    print("=== Spectrum confidence ===")
    print(f"Combined:      max/mean={conf_combined[0]:.6e}, max/2nd={conf_combined[1]:.6e}")
    print(f"LOS only:      max/mean={conf_los[0]:.6e}, max/2nd={conf_los[1]:.6e}")
    print(f"Multipath only:max/mean={conf_mp[0]:.6e}, max/2nd={conf_mp[1]:.6e}")
    print()

    print("=== P min/max ===")
    print(f"Combined:      {np.nanmin(P_combined):.6e} / {np.nanmax(P_combined):.6e}")
    print(f"LOS only:      {np.nanmin(P_los):.6e} / {np.nanmax(P_los):.6e}")
    print(f"Multipath only:{np.nanmin(P_mp):.6e} / {np.nanmax(P_mp):.6e}")
    print()

    print("=== NaN checks ===")
    print(f"Combined:      {np.isnan(P_combined).any()}")
    print(f"LOS only:      {np.isnan(P_los).any()}")
    print(f"Multipath only:{np.isnan(P_mp).any()}")

    print("||v_los||^2 =", np.vdot(v_los, v_los).real)
    print("||v_mp||^2  =", np.vdot(v_mp, v_mp).real)
    print("||v_comb||^2=", np.vdot(v_combined, v_combined).real)

    E_combined = outs["rx_efield"]
    E_los = outs["rx_efield_no_mp"]
    E_mp = E_combined - E_los

    print("||E_los||^2 =", np.vdot(E_los, E_los).real)
    print("||E_mp||^2  =", np.vdot(E_mp, E_mp).real)
    print("||E_comb||^2=", np.vdot(E_combined, E_combined).real)

    print("||v_los||^2 =", np.vdot(v_los, v_los).real)
    print("||v_mp||^2  =", np.vdot(v_mp, v_mp).real)
    print("||v_comb||^2=", np.vdot(v_combined, v_combined).real)

    cross = 2*np.real(np.vdot(v_los, v_mp))
    print("cross term  =", cross)
    print("reconstruct =", np.vdot(v_los, v_los).real + np.vdot(v_mp, v_mp).real + cross)

if False:

    # TX Origin Grid
    X = np.linspace(5, 200, 195)
    Z = np.linspace(5, 110, 105)

    # AOA Ray Grid
    az_fixed_deg = -180.0
    az = np.deg2rad(np.array([az_fixed_deg], dtype=np.float64))  # shape (1,)
    el = np.deg2rad(np.arange(-90.0, 90.0 + 0.25, 0.25))  # fine grid, 0.25°

    true_el_deg = np.zeros((len(Z), len(X)), dtype=np.float64)
    pred_el_deg = np.zeros((len(Z), len(X)), dtype=np.float64)
    el_err_deg  = np.zeros((len(Z), len(X)), dtype=np.float64)

    def wrap_deg_180(a):
        """Wrap degrees to [-180,180)."""
        return (a + 180.0) % 360.0 - 180.0

    i = 0
    for iz, z in enumerate(Z):
        for ix, x in enumerate(X):
            if i % 25==0:
                print(i)
                
            i+=1
            # --- Set TX position
            tx_pos = np.array([x, 0.0, z], dtype=np.float64)
            scene.tx.origin = tx_pos.astype(np.float32)

            # --- Raycast / simulate
            outs, debugs = raycast(scene, rtconstants)
            outs = outs.to_numpy()

            voltages = outs["rx_v"]
            v = voltages[:, 0] + 1j * voltages[:, 1]

            # --- Beamform with fixed azimuth, scan elevation
            P = beamform_power(v, az, el, rx.origin, rx.sub_centers)  # expected shape (len(el), len(az)) == (len(el), 1)
            ie = int(np.argmax(P[:, 0]))
            el_hat = el[ie]

            pred = np.rad2deg(el_hat)

            # --- True elevation of propagation direction TX->RX
            k = scene.rx.origin - tx_pos
            k /= np.linalg.norm(k)
            # elevation = asin(k_z)
            true = np.rad2deg(np.arcsin(np.clip(k[2], -1.0, 1.0)))

            # store
            true_el_deg[iz, ix] = true
            pred_el_deg[iz, ix] = pred
            el_err_deg[iz, ix]  = wrap_deg_180(pred - true)

    # ---- Plot heatmaps
    # Use pcolormesh so axes correspond to X and Z nicely
    import matplotlib.colors as colors
    Xg, Zg = np.meshgrid(X, Z)
    norm = colors.TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)

    plt.figure()
    plt.imshow(
        el_err_deg,
        origin="lower",
        cmap="coolwarm", 
        norm=norm,
        extent=[X.min(), X.max(), Z.min(), Z.max()],
        aspect="auto",
    )

    plt.colorbar(label="Elevation error (deg)")
    plt.gca().invert_xaxis()
    plt.xlabel("Horizontal distance (m)")
    plt.ylabel("Vertical distance (m)")
    plt.title(f"Beamformer elevation error")
    plt.show()

#Beamforming test
if True:

    outs, debugs = rt.raycast(scene, rtconstants)
    outs = outs.to_numpy()

    # Received scalar voltages
    voltages_combined = outs["rx_v"]
    v_combined = voltages_combined[:, 0] + 1j * voltages_combined[:, 1]

    voltages_los = outs["rx_v_no_mp"]
    v_los = voltages_los[:, 0] + 1j * voltages_los[:, 1]

    voltages_mp = voltages_combined - voltages_los
    v_mp = voltages_mp[:, 0] + 1j * voltages_mp[:, 1]

    # Beam spectra
    t0 = time.perf_counter()

    combined = beamformer_wp(voltages_combined, rx.origin, rx.sub_centers, 2.48e9, device)
    los = beamformer_wp(voltages_los, rx.origin, rx.sub_centers, 2.48e9, device)
    mp = beamformer_wp(voltages_mp, rx.origin, rx.sub_centers, 2.48e9, device)

    t1 = time.perf_counter()

    print("3x Beamforming time:", t1 - t0)

    # Peak estimates
    el_hat_combined = combined[2]
    el_hat_los = los[2]
    el_hat_mp = mp[2]

    # True elevation
    tx_pos = scene.tx.origin.astype(np.float64)
    k = scene.rx.origin.astype(np.float64) - tx_pos
    k /= np.linalg.norm(k)
    true_el = np.arctan2(k[2], np.sqrt(k[0] ** 2 + k[1] ** 2))

    # Received energies
    E_combined = np.sum(np.abs(v_combined) ** 2)
    E_los = np.sum(np.abs(v_los) ** 2)
    E_mp = np.sum(np.abs(v_mp) ** 2)

    # Confidence metrics
    def confidence_metrics(P: np.ndarray):
        P = np.ravel(P)
        Pmax = np.max(P)
        Pmean = np.mean(P)
        Psorted = np.sort(P)
        P2 = Psorted[-2] if len(Psorted) > 1 else Pmax
        conf_mean = Pmax / (Pmean + 1e-30)
        conf_second = Pmax / (P2 + 1e-30)
        return conf_mean, conf_second

    conf_combined = confidence_metrics(combined[0])
    conf_los = confidence_metrics(los[0])
    conf_mp = confidence_metrics(mp[0])

    # Plot spectra
    plt.figure(figsize=(10, 6))
    el = np.deg2rad(np.arange(-90.0, 91.0, 1.0))
    plt.plot(np.rad2deg(el), combined[0], label="Combined")
    plt.plot(np.rad2deg(el), los[0], label="LOS only")
    plt.plot(np.rad2deg(el), mp[0], label="Multipath only")

    plt.axvline(np.rad2deg(true_el), color="k", linestyle="--", label="True el")
    plt.axvline(np.rad2deg(el_hat_combined), color="r", linestyle="--", alpha=0.8, label="Combined peak")
    plt.axvline(np.rad2deg(el_hat_los), color="g", linestyle="--", alpha=0.8, label="LOS peak")
    plt.axvline(np.rad2deg(el_hat_mp), color="b", linestyle="--", alpha=0.8, label="MP peak")

    plt.xlabel("Elevation (deg)")
    plt.ylabel("Power")
    plt.title("Beamformer Elevation Spectra")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print diagnostics
    print(f"True elevation: {np.rad2deg(true_el):.3f} deg")
    print()

    print("=== Peak elevations ===")
    print(f"Combined:      {np.rad2deg(el_hat_combined):.3f} deg")
    print(f"LOS only:      {np.rad2deg(el_hat_los):.3f} deg")
    print(f"Multipath only:{np.rad2deg(el_hat_mp):.3f} deg")
    print()

    print("=== Total received energy sum(|v|^2) ===")
    print(f"Combined:      {E_combined:.6e}")
    print(f"LOS only:      {E_los:.6e}")
    print(f"Multipath only:{E_mp:.6e}")
    print(f"MP / LOS:      {E_mp / (E_los + 1e-30):.6e}")
    print()

    print("=== Spectrum confidence ===")
    print(f"Combined:      max/mean={conf_combined[0]:.6e}, max/2nd={conf_combined[1]:.6e}")
    print(f"LOS only:      max/mean={conf_los[0]:.6e}, max/2nd={conf_los[1]:.6e}")
    print(f"Multipath only:max/mean={conf_mp[0]:.6e}, max/2nd={conf_mp[1]:.6e}")
    print()

    print("=== P min/max ===")
    print(f"Combined:      {np.nanmin(combined[0]):.6e} / {np.nanmax(combined[0]):.6e}")
    print(f"LOS only:      {np.nanmin(los[0]):.6e} / {np.nanmax(los[0]):.6e}")
    print(f"Multipath only:{np.nanmin(mp[0]):.6e} / {np.nanmax(mp[0]):.6e}")
    print()

    print("=== NaN checks ===")
    print(f"Combined:      {np.isnan(combined[0]).any()}")
    print(f"LOS only:      {np.isnan(los[0]).any()}")
    print(f"Multipath only:{np.isnan(mp[0]).any()}")

    print("||v_los||^2 =", np.vdot(v_los, v_los).real)
    print("||v_mp||^2  =", np.vdot(v_mp, v_mp).real)
    print("||v_comb||^2=", np.vdot(v_combined, v_combined).real)

    E_combined = outs["rx_efield"]
    E_los = outs["rx_efield_no_mp"]
    E_mp = E_combined - E_los

    print("||E_los||^2 =", np.vdot(E_los, E_los).real)
    print("||E_mp||^2  =", np.vdot(E_mp, E_mp).real)
    print("||E_comb||^2=", np.vdot(E_combined, E_combined).real)

    print("||v_los||^2 =", np.vdot(v_los, v_los).real)
    print("||v_mp||^2  =", np.vdot(v_mp, v_mp).real)
    print("||v_comb||^2=", np.vdot(v_combined, v_combined).real)

    cross = 2*np.real(np.vdot(v_los, v_mp))
    print("cross term  =", cross)
    print("reconstruct =", np.vdot(v_los, v_los).real + np.vdot(v_mp, v_mp).real + cross)

# Two-Wave Beamformer model test:
if True:
    outs, debugs = rt.raycast(scene, rtconstants)
    outs = outs.to_numpy()

    # Received scalar voltages
    voltages_combined = outs["rx_v"]
    v_combined = voltages_combined[:, 0] + 1j * voltages_combined[:, 1]

    voltages_mp = outs["rx_v"] - outs["rx_v_no_mp"] 
    v_mp = voltages_combined[:, 0] + 1j * voltages_combined[:, 1]

    # Beam spectra
    t0 = time.perf_counter()
    P_mp, _, _ = beamformer_wp(voltages_mp, rx.origin, rx.sub_centers, 2.48e9, device)
    P_comb, _, _ = beamformer_wp(voltages_combined, rx.origin, rx.sub_centers, 2.48e9, device)
    t1 = time.perf_counter()

    # True elevation
    tx_pos = scene.tx.origin.astype(np.float64)
    k = scene.rx.origin.astype(np.float64) - tx_pos
    k /= np.linalg.norm(k)
    true_el = np.arctan2(k[2], np.sqrt(k[0] ** 2 + k[1] ** 2))

    # Received energies
    E_combined = np.sum(np.abs(v_combined) ** 2)

    # Confidence metrics
    def confidence_metrics(P: np.ndarray):
        P = np.ravel(P)
        Pmax = np.max(P)
        Pmean = np.mean(P)
        Psorted = np.sort(P)
        P2 = Psorted[-2] if len(Psorted) > 1 else Pmax
        conf_mean = Pmax / (Pmean + 1e-30)
        conf_second = Pmax / (P2 + 1e-30)
        return conf_mean, conf_second

    # Plot spectra
    plt.figure(figsize=(10, 6))
    plt.yscale("log")
    el = np.deg2rad(np.arange(-90.0, 90.0 + 0.25, 0.25))
    plt.plot(np.rad2deg(el), P1, label="Dual wave 1st")
    plt.plot(np.rad2deg(el), P2, label="Dual wave 2nd")
    plt.plot(np.rad2deg(el), P_comb, label="Combined")
    plt.plot(np.rad2deg(el), P_mp, label="Multipath")

    plt.axvline(np.rad2deg(true_el), color="k", linestyle="--", label="True el")
    plt.axvline(np.rad2deg(el_hat_1), color="r", linestyle="--", alpha=0.8, label="Wave 1")
    plt.axvline(np.rad2deg(el_hat_2), color="b", linestyle="--", alpha=0.8, label="Wave 2")

    plt.xlabel("Elevation (deg)")
    plt.ylabel("Power")
    plt.title("Beamformer Elevation Spectra")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Print diagnostics
    print(f"True elevation: {np.rad2deg(true_el):.3f} deg")
    print()

    print("=== Peak elevations ===")
    print(f"Wave 1:      {np.rad2deg(el_hat_1):.3f} deg")
    print(f"Wave 2:      {np.rad2deg(el_hat_2):.3f} deg")
    print()

    print("=== Total received energy sum(|v|^2) ===")
    print(f"Combined:      {E_combined:.6e}")
    print()

    print("=== Spectrum confidence ===")
    print(f"Combined:      max/mean={conf_combined[0]:.6e}, max/2nd={conf_combined[1]:.6e}")
    print()

    print("=== P min/max ===")
    print(f"Combined:      {np.nanmin(P1):.6e} / {np.nanmax(P1):.6e}")
    print()

    print("=== NaN checks ===")
    print(f"Combined:      {np.isnan(P1).any()}")

    print("||v_comb||^2=", np.vdot(v_combined, v_combined).real)

    E_combined = outs["rx_efield"]

    print("||E_comb||^2=", np.vdot(E_combined, E_combined).real)

    print("||v_comb||^2=", np.vdot(v_combined, v_combined).real)


# Comparison against real data
if False:

    data = pd.read_csv("20221210_data.csv")
    
    errors = []
    dist = []
    height = []

    for burst_id, b in data.groupby("burst_id"):

        f_cfo = b["f_cfo"].iloc[0]
        x = b["s"].median()
        y = 0
        z = max(b["z"].median(), 0.1)
        scene.freq = 2.48e9 + f_cfo
        scene.tx.origin = np.array([x,y,z])

        phases = np.zeros(12)

        for ant_id, a in b.groupby("antenna_id"):
            phases[ant_id-1] = a["phi_corr"].mean()

        outs, debugs = rt.raycast(scene, rtconstants)
        outs = outs.to_numpy()
        debugs = debugs.to_numpy()

        real = phases - phases[0]
        sim = np.atan2(outs["rx_v"][:, 1],outs["rx_v"][:, 0]) - np.atan2(outs["rx_v"][0, 1],outs["rx_v"][0, 0])

        error = np.angle(np.exp(1j * (real-sim)))
        error = np.mean(error**2)
        errors.append(error)

        height.append(z)
        dist.append(x)


    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(errors)
    axs[0].set_ylabel("Phase MSE [rad^2]")
    axs[0].set_title("Phase error MSE")
    axs[0].grid(True)

    axs[1].plot(dist)
    axs[1].set_xlabel("burst_index")
    axs[1].set_ylabel("xy distance [m]")
    axs[1].set_title("xy distance [m]")
    axs[1].grid(True)

    axs[2].plot(height)
    axs[2].set_xlabel("burst_index")
    axs[2].set_ylabel("Height [m]")
    axs[2].set_title("Height")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
            