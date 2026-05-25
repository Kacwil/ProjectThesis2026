import numpy as np
from utils import *
import warp as wp


C = 299_792_458.0

@wp.func
def los_vector(az: wp.float32, el: wp.float32):
    """
    Returns line of sight vector in antenna frame.
    
    """

    se = wp.sin(el)
    ce = wp.cos(el)
    ca = wp.cos(az)
    sa = wp.sin(az)
    return wp.vec3(se * ca, se * sa, ce)


@wp.kernel
def music_spectrum_kernel(
    ant_pos: wp.array(dtype=wp.vec3),       # (N,)
    en_re: wp.array2d(dtype=wp.float32),    # (N, Nn)
    en_im: wp.array2d(dtype=wp.float32),    # (N, Nn)
    azi_angles: wp.array(dtype=wp.float32), # (Naz,)
    ele_angles: wp.array(dtype=wp.float32), # (Nel,)
    wavelength: wp.float32,
    results: wp.array(dtype=wp.float32),    # (Naz * Nel,)
):
    tid = wp.tid()

    n_az = azi_angles.shape[0]
    n_el = ele_angles.shape[0]

    ia = tid // n_el
    ie = tid - ia * n_el

    if ia >= n_az or ie >= n_el:
        return

    az = azi_angles[ia]
    el = ele_angles[ie]

    los = los_vector(az, el)

    n_ant = ant_pos.shape[0]
    n_noise = en_re.shape[1]

    # denom = ||En^H a||^2
    # where a_k = exp(j*phase_k)
    # and [En^H a]_m = sum_k conj(En[k,m]) * a_k
    denom = wp.float32(0.0)

    for m in range(n_noise):
        proj_re = wp.float32(0.0)
        proj_im = wp.float32(0.0)

        for k in range(n_ant):
            r = ant_pos[k]
            phase = 2.0 * wp.pi * wp.dot(los, r) / wavelength

            a_re = wp.cos(phase)
            a_im = wp.sin(phase)

            # conj(En[k,m]) = en_re[k,m] - j en_im[k,m]
            er = en_re[k, m]
            ei = en_im[k, m]

            # conj(En[k,m]) * a_k
            # real = er*a_re + ei*a_im
            # imag = er*a_im - ei*a_re
            proj_re += er * a_re + ei * a_im
            proj_im += er * a_im - ei * a_re

        denom += proj_re * proj_re + proj_im * proj_im

    if denom < 1.0e-12:
        denom = 1.0e-12

    results[tid] = 1.0 / denom

def music_spectrum_wp(
    x,
    ant_pos,
    wavelength,
    az_grid_deg,
    el_grid_deg,
    n_sources=1,
    diagonal_loading=1e-6,
    device="cuda",
    debug=False,
):
    import time

    t0_total = time.perf_counter()

    x = np.asarray(x, dtype=np.complex64)
    ant_pos = np.asarray(ant_pos, dtype=np.float32)

    # ------------------------------------------------------------
    # Accept either:
    #   x shape (N,)      single snapshot
    #   x shape (T, N)    T snapshots, N antennas
    #   x shape (N, T)    N antennas, T snapshots
    # Internally use X as shape (N, T)
    # ------------------------------------------------------------
    if x.ndim == 1:
        X = x.reshape(-1, 1)          # (N, 1)
    elif x.ndim == 2:
        # Prefer burst_mat convention: (T, N)
        if x.shape[1] == ant_pos.shape[0]:
            X = x.T                   # (N, T)
        elif x.shape[0] == ant_pos.shape[0]:
            X = x                     # already (N, T)
        else:
            raise ValueError(
                f"x shape {x.shape} is incompatible with ant_pos shape {ant_pos.shape}"
            )
    else:
        raise ValueError(f"x must be 1D or 2D, got shape {x.shape}")

    n_ant, n_snapshots = X.shape

    if ant_pos.shape != (n_ant, 3):
        raise ValueError(f"Expected ant_pos shape {(n_ant, 3)}, got {ant_pos.shape}")

    if not (1 <= n_sources < n_ant):
        raise ValueError(f"n_sources must satisfy 1 <= n_sources < {n_ant}")

    # ------------------------------------------------------------------
    # Covariance + eigendecomposition on CPU
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    R = (X @ X.conj().T) / max(n_snapshots, 1)
    R = R.astype(np.complex64)

    if diagonal_loading > 0.0:
        load = diagonal_loading * np.trace(R).real / n_ant
        R = R + load * np.eye(n_ant, dtype=np.complex64)

    eigvals, eigvecs = np.linalg.eigh(R)

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    En = eigvecs[:, : n_ant - n_sources].astype(np.complex64)

    t1 = time.perf_counter()
    if debug:
        print(f"[music-wp] eigensolve + En prep : {t1 - t0:.4f} s")
        print(f"[music-wp] n_ant={n_ant}, n_snapshots={n_snapshots}, n_noise={En.shape[1]}")

    # ------------------------------------------------------------------
    # Grid prep
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    az_grid_deg = np.asarray(az_grid_deg, dtype=np.float32)
    el_grid_deg = np.asarray(el_grid_deg, dtype=np.float32)

    az_rad = np.deg2rad(az_grid_deg).astype(np.float32)
    el_rad = np.deg2rad(el_grid_deg).astype(np.float32)

    n_az = az_rad.shape[0]
    n_el = el_rad.shape[0]

    t1 = time.perf_counter()
    if debug:
        print(f"[music-wp] grid prep            : {t1 - t0:.4f} s")
        print(f"[music-wp] grid size            : {n_az} x {n_el} = {n_az * n_el}")

    # ------------------------------------------------------------------
    # Warp arrays
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    ant_wp = wp.array(ant_pos, dtype=wp.vec3, device=device)
    en_re_wp = wp.array2d(np.real(En).astype(np.float32), dtype=wp.float32, device=device)
    en_im_wp = wp.array2d(np.imag(En).astype(np.float32), dtype=wp.float32, device=device)
    az_wp = wp.array(az_rad, dtype=wp.float32, device=device)
    el_wp = wp.array(el_rad, dtype=wp.float32, device=device)
    results_wp = wp.zeros(n_az * n_el, dtype=wp.float32, device=device)

    t1 = time.perf_counter()
    if debug:
        print(f"[music-wp] device transfer      : {t1 - t0:.4f} s")

    # ------------------------------------------------------------------
    # Kernel launch
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    wp.launch(
        kernel=music_spectrum_kernel,
        dim=n_az * n_el,
        inputs=[
            ant_wp,
            en_re_wp,
            en_im_wp,
            az_wp,
            el_wp,
            np.float32(wavelength),
            results_wp,
        ],
        device=device,
    )

    P_music = results_wp.numpy().reshape(n_az, n_el)

    t1 = time.perf_counter()
    if debug:
        print(f"[music-wp] kernel + copy back   : {t1 - t0:.4f} s")
        print(f"[music-wp] total                : {time.perf_counter() - t0_total:.4f} s")

    return P_music

def music_top_minimal_wp(
    burst_mat,
    ant_pos_ordered=None,
    fr_mhz=None,
    device="cuda",
    az_step=0.5,
    el_step=0.5,
    n_sources=1,
    diagonal_loading=1e-6,
    debug=False,
    ant_pos=None,
    use_ref_ant=True,
    ref_ant_col=10,   # antenna 11 if columns are [1, 2, ..., 12]
):
    X = np.asarray(burst_mat, dtype=np.complex64)

    if X.ndim != 2:
        raise ValueError(f"burst_mat must be 2D, got shape {X.shape}")

    n_meas = X.shape[-1]

    if n_meas == 82:
        if ant_pos_ordered is None:
            raise ValueError("For raw burst_mat shape (T, 82), ant_pos_ordered is required")
        pos_in = ant_pos_ordered
        expected_n = 82
        pos_name = "ant_pos_ordered"

    elif n_meas == 12:
        if ant_pos is None:
            raise ValueError("For per-antenna averaged burst_mat shape (T, 12), ant_pos is required")
        pos_in = ant_pos
        expected_n = 12
        pos_name = "ant_pos"

    else:
        raise ValueError(
            f"Unsupported burst_mat antenna length {n_meas}. Expected 82 raw channels or 12 averaged antennas."
        )

    pos_in = np.asarray(pos_in, dtype=np.float32)

    if pos_in.ndim != 2:
        raise ValueError(f"{pos_name} must be 2D")

    if pos_in.shape == (3, expected_n):
        ant_pos_np = pos_in.T.copy()
    elif pos_in.shape == (expected_n, 3):
        ant_pos_np = pos_in.copy()
    else:
        raise ValueError(
            f"{pos_name} shape {pos_in.shape} is incompatible with burst_mat antenna length {n_meas}; "
            f"expected (3, {expected_n}) or ({expected_n}, 3)"
        )

    # ------------------------------------------------------------
    # Optionally remove reference antenna
    # ------------------------------------------------------------
    if not use_ref_ant:
        if n_meas != 12:
            raise ValueError(
                "use_ref_ant=False is currently only supported for burst_mat shape (T, 12). "
                "For raw (T, 82), remove reference samples before calling this function."
            )

        keep = np.ones(X.shape[1], dtype=bool)
        keep[ref_ant_col] = False

        X = X[:, keep]                  # (T, 11)
        ant_pos_np = ant_pos_np[keep]   # (11, 3)

    if fr_mhz is None:
        raise ValueError("fr_mhz is required")

    wavelength = C / fr_mhz

    az_grid = np.arange(-180.0, 180.0 + 1e-9, az_step, dtype=np.float32)
    el_grid = np.arange(0.0, 90.0 + 1e-9, el_step, dtype=np.float32)

    P = music_spectrum_wp(
        x=X,
        ant_pos=ant_pos_np,
        wavelength=wavelength,
        az_grid_deg=az_grid,
        el_grid_deg=el_grid,
        n_sources=n_sources,
        diagonal_loading=diagonal_loading,
        device=device,
        debug=debug,
    )

    ia_max, ie_max = np.unravel_index(np.argmax(P), P.shape)

    return {
        "P": P,
        "az_grid": az_grid,
        "el_grid": el_grid,
        "az_hat_deg": float(az_grid[ia_max]),
        "el_hat_deg": float(el_grid[ie_max]),
        "X": X,
        "ant_pos_used": ant_pos_np,
        "use_ref_ant": use_ref_ant,
    }