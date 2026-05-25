import numpy as np
from utils import Rzyx
import matplotlib.pyplot as plt

def neu_to_ned(v_neu):
    v_neu = np.asarray(v_neu)
    v_ned = v_neu.copy()
    v_ned[..., 2] *= -1.0
    return v_ned

def ned_to_neu(v_ned):
    v_ned = np.asarray(v_ned)
    v_neu = v_ned.copy()
    v_neu[..., 2] *= -1.0
    return v_neu

def az_el_from_ned(v_ned):
    N = v_ned[..., 0]
    E = v_ned[..., 1]
    D = v_ned[..., 2]

    horiz = np.sqrt(N**2 + E**2)

    az = np.arctan2(E, N)
    el = np.arctan2(-D, horiz)

    return az, el

def R_ned_from_ant(boresight_az):

    z_ant_ned = np.array([
        np.cos(boresight_az),   # North
        np.sin(boresight_az),   # East
        0.0                     # Down
    ])

    x_ant_ned = np.array([0.0, 0.0, -1.0])
    y_ant_ned = np.cross(z_ant_ned, x_ant_ned)
    y_ant_ned /= np.linalg.norm(y_ant_ned)

    R_ned_ant = np.column_stack([
        x_ant_ned,
        y_ant_ned,
        z_ant_ned
    ])

    return R_ned_ant

def build_antenna_positions_in_antenna_frame(phi_ant=0.0, theta_ant=0.0, psi_ant=-45.0):

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

    return Rzyx(0.0, 0.0, np.deg2rad(psi_ant)) @ ant_pos

def get_NED_antenna_pos(boresight_az, rx_height=0.42, inspect=False):

    ant_pos_ant = build_antenna_positions_in_antenna_frame()  # (3, 12)

    R_ned_ant = R_ned_from_ant(boresight_az)

    ant_pos_ned = R_ned_ant @ ant_pos_ant

    # RX is physically above ground, so in NED this is negative D
    ant_pos_ned[2, :] -= rx_height

    if inspect:
        ant_pos_neu = ned_to_neu(ant_pos_ned)

        N_ant = ant_pos_neu[0, :]
        E_ant = ant_pos_neu[1, :]

        plt.figure()
        plt.scatter(E_ant, N_ant, label="Antenna elements")

        for i in range(len(E_ant)):
            plt.text(E_ant[i], N_ant[i], str(i))

        N_bore = np.cos(boresight_az)
        E_bore = np.sin(boresight_az)

        plt.quiver(
            0, 0,
            E_bore, N_bore,
            angles="xy",
            scale_units="xy",
            scale=1,
            label="Boresight"
        )

        plt.xlabel("East")
        plt.ylabel("North")
        plt.title("Antenna Geometry in Horizontal Plane")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()

    return ant_pos_ned