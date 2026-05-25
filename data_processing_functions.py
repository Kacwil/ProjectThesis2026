import numpy as np
import pandas as pd
from utils import Rzyx

def process_nav(nav):

    array_gnss_offset = 0.35
    copter_gnss_offset = 0.2
    DOWN_OFFSET = -(array_gnss_offset-copter_gnss_offset)

    ts = nav["timestamp"]
    N = (1e-2 * nav["relPosN"] + 1e-4 * nav["relPosHPN"]).to_numpy()
    E = (1e-2 * nav["relPosE"] + 1e-4 * nav["relPosHPE"]).to_numpy()
    D = (1e-2 * nav["relPosD"] + 1e-4 * nav["relPosHPD"] + DOWN_OFFSET).to_numpy()

    NED = np.array([N, E, D])
    NED = Rzyx(0, 0, np.deg2rad(49.5 + 4.5)) @ NED

    return pd.DataFrame({
        "ts":pd.to_datetime(ts, unit="s", utc=True),
        "N":NED[0, :],
        "E":NED[1, :],
        "D":NED[2, :],
        "dist_NE":  np.sqrt(N**2 + E**2),
        "dist_NED": np.sqrt(N**2 + E**2 + D**2),
        "elevation": np.atan2(-D, np.sqrt(N**2 + E**2)),
        "azimuth": np.atan2(E, N) + np.deg2rad(49.0),
    }).set_index("ts").sort_index()

def prepare_rx(rx):
    rx = rx["nordic"]

    ts = rx["timestamp"] # N
    dt = rx["time"] # N
    ant_idx = rx["antenna_id"] # N, 156
    Q = rx["Q"] #N, 156
    I = rx["I"] #N, 156

    N, M = I.shape
    df = pd.DataFrame({
        "ts":  pd.to_datetime(np.repeat(ts, M) + dt.flatten() * 10**(-6), unit="s", utc=True),
        "burst_id": np.repeat(np.arange(N), M),
        "burst_rel_t": dt.flatten() * (10/8) * 10**(-7),
        "antenna_id": ant_idx.flatten(),
        "I": I.flatten(),
        "Q": Q.flatten(),
        "phi": np.atan2(Q, I).flatten(),
        "amp": np.hypot(I, Q).flatten(),
    })

    return df

def estimate_cfo_per_burst(rx, f_min=230e3, f_max=270e3, n_grid=2001):

    rx = rx.copy()
    rx["x"] = rx["I"].astype(np.float64) + 1j * rx["Q"].astype(np.float64)

    freqs = np.linspace(f_min, f_max, n_grid)
    cfos = {}
    spectrums = []

    for burst_id, h in rx.groupby("burst_id"):

        spectrum = np.zeros_like(freqs, dtype=np.float64)
        for ant_id, g in h.groupby("antenna_id"):

            dt = g["burst_rel_t"].to_numpy()
            x = g["x"].to_numpy()
            A = np.exp(1j * 2 * np.pi * np.outer(freqs, dt))
            corr = A.conj() @ x
            spectrum += np.abs(corr) ** 2

        f_hat = freqs[np.argmax(spectrum)]

        cfos[burst_id] = f_hat
        spectrums.append(spectrum)

    return cfos, spectrums, freqs

def apply_cfo_correction(rx, f_cfos):
    rx = rx.copy()
    rx["f_cfo"] = rx["burst_id"].map(f_cfos).astype(np.float64)
    rx["x"] = rx["I"].astype(np.float64) + 1j * rx["Q"].astype(np.float64)
    t = rx["burst_rel_t"].to_numpy(dtype=np.float64)

    rx["x_corr"] = rx["x"] * np.exp(-1j * 2 * np.pi * rx["f_cfo"].to_numpy() * t)
    rx["I_corr"] = np.real(rx["x_corr"])
    rx["Q_corr"] = np.imag(rx["x_corr"])
    rx["phi_corr"] = np.angle(rx["x_corr"])
    rx["amp_corr"] = np.abs(rx["x_corr"])

    return rx

def get_sampling_order():
    sampling_order = np.array([
        11,11,11,11,11,11,11,11,
        12,1,2,10,3,9,4,8,7,6,5,
        12,1,2,10,3,9,4,8,7,6,5,
        12,1,2,10,3,9,4,8,7,6,5,
        12,1,2,10,3,9,4,8,7,6,5,
        12,1,2,10,3,9,4,8,7,6,5,
        12,1,2,10,3,9,4,8,7,6,5,
        12,1,2,10,3,9,4,8
    ], dtype=int)
    return sampling_order
    

def build_ant_pos():
    ant_pos = np.array([
        [ 0.00, -0.10, 0.0],
        [ 0.00, -0.15, 0.0],
        [-0.05, -0.15, 0.0],
        [-0.10, -0.15, 0.0],
        [-0.15, -0.15, 0.0],
        [-0.15, -0.10, 0.0],
        [-0.15, -0.05, 0.0],
        [-0.15,  0.00, 0.0],
        [-0.10,  0.00, 0.0],
        [-0.05,  0.00, 0.0],
        [ 0.00,  0.00, 0.0],
        [ 0.00, -0.05, 0.0],
    ], dtype=float).T

    return Rzyx(0.0, 0.0, np.deg2rad(-45.0)) @ ant_pos

def build_ant_pos_ordered():
    sampling_order = get_sampling_order()
    ant_pos = build_ant_pos()
    ant_pos_ordered = ant_pos[:, sampling_order - 1]
    return sampling_order, ant_pos_ordered

def interpolate_nav_at_time(nav, time):
    """
    Linearly interpolate two NAV values when given antenna measurement in between.
    """
    el_max, el_min = np.deg2rad(75), np.deg2rad(5)

    idx_next = nav.index.searchsorted(time)

    if idx_next == 0 or idx_next >= len(nav):
        return None

    idx_prev = idx_next - 1

    t0 = nav.index[idx_prev]
    t1 = nav.index[idx_next]

    nav0 = nav.iloc[idx_prev]
    nav1 = nav.iloc[idx_next]


    if not (el_min <= nav0["elevation"] <= el_max):
        return None

    if not (el_min <= nav1["elevation"] <= el_max):
        return None

    alpha = (time - t0) / (t1 - t0)

    burst_nav = nav0 + alpha * (nav1 - nav0)
    burst_nav.name = time

    return burst_nav

    return burst_nav

def prepare_bursts(nav, rx):
    
    out = {}
    sampling_order, ant_pos_ordered = build_ant_pos_ordered()

    for burst_id, burst, in rx.groupby("burst_id"):

        burst = burst.sort_values("burst_rel_t").reset_index(drop=True)
        time = burst["ts"].mean()

        burst_nav = interpolate_nav_at_time(nav, time)

        if burst_nav is None:
            continue

        # Match measurements to the measuring antenna at time t:
        X = np.zeros(82, dtype=np.complex128)
        X_raw = np.zeros(82, dtype=np.complex128)

        sums = np.zeros(12, dtype=np.complex128)
        counts = np.zeros(12, dtype=int)

        I = burst["I"].to_numpy()
        Q = burst["Q"].to_numpy()
        t = burst["burst_rel_t"].to_numpy()
        f_cfo = burst["f_cfo"].iloc[0]

        # first 8 reference samples
        for ref_i in range(8):
            ant_i = sampling_order[ref_i] - 1  # should be 11

            x = I[ref_i] + 1j * Q[ref_i]

            X[ref_i] = x * np.exp(-1j * 2*np.pi * f_cfo * t[ref_i])
            X_raw[ref_i] = x

            sums[ant_i] += X[ref_i]
            counts[ant_i] += 1


        # rest 74
        for switch_i in range(74):
            out_i = 8 + switch_i
            measurement_i = 8 + 2*switch_i
            ant_i = sampling_order[out_i] - 1

            if I[measurement_i] == -32768 or Q[measurement_i] == -32768:
                X[out_i] = 0.0
                X_raw[out_i] = 0.0
            else:
                x = I[measurement_i] + 1j * Q[measurement_i]

                X[out_i] = x * np.exp(-1j * 2*np.pi * f_cfo * t[measurement_i])
                X_raw[out_i] = x

            sums[ant_i] += X[out_i]
            counts[ant_i] += 1

        X_avg = np.zeros(12, dtype=np.complex128)
        valid = counts > 0
        X_avg[valid] = sums[valid] / counts[valid]

        out[burst_id] = {  
            "MATCHED": {
                "X":X,
                "X_RAW":X_raw,
                "X_AVG":X_avg,
            },

            "FULL": {
                "I": I,
                "Q": Q,
                "TS": burst["ts"].to_numpy(),
                "TIME": t,
                "ANTENNA_ID": burst["antenna_id"].to_numpy(),
            },  

            "NAV": burst_nav,
            "CFO": f_cfo,
            "BURST_TS": time,}

    data = {"DATA": out,
            "ANT_POS_ORDERED": ant_pos_ordered}
        
    return data
