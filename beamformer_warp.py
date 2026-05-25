import warp as wp
import numpy as np

C = 299_792_458.0

@wp.func
def steering_vector(az: wp.float32, el: wp.float32):
    se = wp.sin(el)
    ce = wp.cos(el)
    ca = wp.cos(az)
    sa = wp.sin(az)
    return wp.vec3(se * ca, se * sa, ce)

@wp.kernel
def beamform_matlab_kernel(
    x_re: wp.array(dtype=wp.float32),          # (N,)
    x_im: wp.array(dtype=wp.float32),          # (N,)
    ant_pos: wp.array(dtype=wp.vec3),          # (N,)
    azi_angles: wp.array(dtype=wp.float32),    # (Naz,)
    ele_angles: wp.array(dtype=wp.float32),    # (Nel,)
    wavelength: wp.float32,
    results: wp.array(dtype=wp.float32),       # (Naz * Nel,)
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

    los = steering_vector(az, el)

    acc_re = wp.float32(0.0)
    acc_im = wp.float32(0.0)

    n = x_re.shape[0]

    for k in range(n):
        r = ant_pos[k]
        phase = 2.0 * wp.pi * wp.dot(los, r) / wavelength

        a_re = wp.cos(phase)
        a_im = wp.sin(phase)

        xr = x_re[k]
        xi = x_im[k]

        acc_re += a_re * xr + a_im * xi
        acc_im += a_re * xi - a_im * xr

    power = acc_re * acc_re + acc_im * acc_im
    results[tid] = power


# ------------------------------------------------------------------
# CPU helpers for local refinement
# ------------------------------------------------------------------

def steering_vector_np(ant_pos, az_rad, el_rad, wavelength):
    """
    ant_pos: (N, 3)
    returns steering vector a: (N,)
    """
    se = np.sin(el_rad)
    ce = np.cos(el_rad)
    ca = np.cos(az_rad)
    sa = np.sin(az_rad)

    los = np.array([se * ca, se * sa, ce], dtype=np.float64)
    phase = 2.0 * np.pi * (ant_pos @ los) / wavelength
    return np.exp(1j * phase)


def beamformer_power_np(az_deg, el_deg, x, ant_pos, wavelength):
    """
    Continuous beamformer objective on CPU.
    """
    az_rad = np.deg2rad(az_deg)
    el_rad = np.deg2rad(el_deg)

    a = steering_vector_np(ant_pos, az_rad, el_rad, wavelength)
    y = np.vdot(a, x)   # conj(a)^T x
    return np.abs(y) ** 2


def refine_beamformer_peak(
    x,
    ant_pos,
    wavelength,
    az_init_deg,
    el_init_deg,
    az_window_deg=2.0,
    el_window_deg=2.0,
    method="Powell",
):
    """
    Local CPU refinement around the coarse Warp peak.

    Returns
    -------
    az_refined_deg, el_refined_deg, success, fun
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Fallback: return coarse estimate unchanged
        return az_init_deg, el_init_deg, False, None

    def objective(z):
        az_deg, el_deg = z

        # keep search inside physical bounds
        if el_deg < 0.0 or el_deg > 90.0:
            return 1e30

        return -beamformer_power_np(
            az_deg=az_deg,
            el_deg=el_deg,
            x=x,
            ant_pos=ant_pos,
            wavelength=wavelength,
        )

    bounds = [
        (az_init_deg - az_window_deg, az_init_deg + az_window_deg),
        (max(0.0, el_init_deg - el_window_deg), min(90.0, el_init_deg + el_window_deg)),
    ]

    result = minimize(
        objective,
        x0=np.array([az_init_deg, el_init_deg], dtype=np.float64),
        method=method,
        bounds=bounds if method in {"L-BFGS-B", "TNC", "SLSQP", "Powell"} else None,
        options={"maxiter": 100},
    )

    az_refined = float(result.x[0])
    el_refined = float(result.x[1])

    return az_refined, el_refined, bool(result.success), float(result.fun)


# ------------------------------------------------------------------
# Host wrapper
# ------------------------------------------------------------------

def beamform_top_minimal_wp(
    burst,
    ant_pos_ordered,
    fr_mhz,
    device="cpu",
    az_step=1.0,
    el_step=1.0,
    refine=True,
    refine_method="Powell",
    debug=False,
):
    
    X = np.asarray(burst["MATCHED"]["X"], dtype=np.complex64)
    ant_pos_ordered = np.asarray(ant_pos_ordered, dtype=np.float32)

    if ant_pos_ordered.ndim != 2:
        raise ValueError("ant_pos_ordered must be 2D")

    # accept either (N, 3) or (3, N)
    if ant_pos_ordered.shape[1] == 3 and ant_pos_ordered.shape[0] == X.shape[0]:
        ant_pos_np = ant_pos_ordered.copy()
    elif ant_pos_ordered.shape[0] == 3 and ant_pos_ordered.shape[1] == X.shape[0]:
        ant_pos_np = ant_pos_ordered.T.copy()
    else:
        raise ValueError(
            f"ant_pos_ordered shape {ant_pos_ordered.shape} incompatible with X length {X.shape[0]}"
        )

    wavelength = C / (fr_mhz * 1e6 + float(burst["CFO"]))

    az_grid = np.arange(-180.0, 180.0 + 1e-9, az_step, dtype=np.float32)
    el_grid = np.arange(0.0, 90.0 + 1e-9, el_step, dtype=np.float32)

    az_rad = np.deg2rad(az_grid).astype(np.float32)
    el_rad = np.deg2rad(el_grid).astype(np.float32)

    x_re_wp = wp.array(np.real(X).astype(np.float32), dtype=wp.float32, device=device)
    x_im_wp = wp.array(np.imag(X).astype(np.float32), dtype=wp.float32, device=device)
    ant_wp = wp.array(ant_pos_np, dtype=wp.vec3, device=device)
    az_wp = wp.array(az_rad, dtype=wp.float32, device=device)
    el_wp = wp.array(el_rad, dtype=wp.float32, device=device)

    n_az = az_grid.shape[0]
    n_el = el_grid.shape[0]
    results_wp = wp.zeros(n_az * n_el, dtype=wp.float32, device=device)

    wp.launch(
        kernel=beamform_matlab_kernel,
        dim=n_az * n_el,
        inputs=[
            x_re_wp,
            x_im_wp,
            ant_wp,
            az_wp,
            el_wp,
            np.float32(wavelength),
            results_wp,
        ],
        device=device,
    )

    P = results_wp.numpy().reshape(n_az, n_el)

    ia_max, ie_max = np.unravel_index(np.argmax(P), P.shape)
    az_hat_grid = float(az_grid[ia_max])
    el_hat_grid = float(el_grid[ie_max])

    az_hat = az_hat_grid
    el_hat = el_hat_grid
    refine_success = False
    refine_fun = None

    if refine:
        az_hat, el_hat, refine_success, refine_fun = refine_beamformer_peak(
            x=X,
            ant_pos=ant_pos_np,
            wavelength=wavelength,
            az_init_deg=az_hat_grid,
            el_init_deg=el_hat_grid,
            az_window_deg=az_step/2,
            el_window_deg=el_step/2,
            method=refine_method,
        )

        if debug:
            print(
                f"[refine] grid=({az_hat_grid:.3f}, {el_hat_grid:.3f}) deg -> "
                f"refined=({az_hat:.3f}, {el_hat:.3f}) deg | "
                f"success={refine_success}"
            )

    return {
        "P": P,
        "az_grid": az_grid,
        "el_grid": el_grid,
        "az_hat_deg": az_hat,
        "el_hat_deg": el_hat,
        "az_hat_grid_deg": az_hat_grid,
        "el_hat_grid_deg": el_hat_grid,
        "refine_success": refine_success,
        "refine_fun": refine_fun,
        "X": X,
    }