import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pickle
import pandas as pd


def normalize(v: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.zeros_like(v)
    return v / norm

def reflect_2D(n, d):
    return d - 2.0 * np.dot(d, n) * n

def los_vector(az, el):
    """
    Returns line of sight vector in antenna frame.
    """

    se = np.sin(el)
    ce = np.cos(el)
    ca = np.cos(az)
    sa = np.sin(az)
    return np.array([se * ca, se * sa, ce])

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

def get_NEU_antenna_pos(boresight_az, inspect=False):
    ant_pos = build_ant_pos() # 3, 12

    z_ant_ned = np.array([
        np.cos(boresight_az),   # North
        np.sin(boresight_az),   # East
        0.0                     # Down
    ])

    x_ant_ned = np.array([0.0, 0.0, -1.0])  # antenna x-axis = world Up = -Down

    y_ant_ned = np.cross(z_ant_ned, x_ant_ned)
    y_ant_ned /= np.linalg.norm(y_ant_ned)

    R_ned_ant = np.column_stack([
        x_ant_ned,
        y_ant_ned,
        z_ant_ned
    ])

    ant_offsets_ned = R_ned_ant @ ant_pos
    ant_offsets_ned[2, :] -= 0.42  # +0.42 Up means -0.42 Down

    N_ant = ant_offsets_ned[0, :]
    E_ant = ant_offsets_ned[1, :]

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
        plt.show()


    return ant_offsets_ned


def steering_vector(ant_pos, az_rad, el_rad, wavelength):

    se = np.sin(el_rad)
    ce = np.cos(el_rad)
    ca = np.cos(az_rad)
    sa = np.sin(az_rad)

    los = np.array([se * ca, se * sa, ce], dtype=np.float64)
    phase = 2.0 * np.pi * (ant_pos @ los) / wavelength

    return np.exp(1j * phase).astype(np.complex64)




def Rzyx(phi, theta, psi):
    """
    phi=roll / (x), theta=pitch (y), psi=yaw (z).
    """
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cth  = np.cos(theta)
    sth  = np.sin(theta)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)

    R = np.array([
        [cpsi*cth, -spsi*cphi + cpsi*sth*sphi,  spsi*sphi + cpsi*cphi*sth],
        [spsi*cth,  cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi],
        [-sth,      cth*sphi,                   cth*cphi]
    ])
    return R

def wrap180(angle_deg):
    return (angle_deg + 180.0) % 360.0 - 180.0


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

def load_iq_dataset():

    with open(r"data\experimental_data.pkl", "rb") as f:
        raw = pickle.load(f)

    ant_pos = raw["ANT_POS_ORDERED"].T
    data = raw["DATA"]

    keys = sorted(data.keys())
    n = len(keys)

    iq = np.zeros((n, 82, 2), dtype=np.float32)
    angles = np.zeros((n, 2), dtype=np.float32)
    positions = np.zeros((n, 2), dtype=np.float32)
    timestamps = []

    for i, burst_id in enumerate(keys):
        burst = data[burst_id]
        x = burst["MATCHED"]["X_RAW"]

        iq[i, :, 0] = np.real(x)
        iq[i, :, 1] = np.imag(x)

        angles[i, 0] = burst["NAV"]["azimuth"]
        angles[i, 1] = burst["NAV"]["elevation"]

        positions[i, 0] = burst["NAV"]["dist_NE"]
        positions[i, 1] = burst["NAV"]["D"]

        timestamps.append(data[burst_id]["BURST_TS"])

    timestamps = pd.to_datetime(timestamps)
    mean_dt = (
        timestamps.to_series()
        .diff()
        .dt.total_seconds()
        .mean()
    )

    return iq, ant_pos.astype(np.float32), angles, positions, keys, mean_dt


