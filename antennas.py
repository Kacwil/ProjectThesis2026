import warp as wp
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from utils import Rzyx
from raytracer.utils_rt import wp_Rzyx


class Transmitter:
    def __init__(self, origin, orientation=np.array([0, np.deg2rad(10), np.pi])):

        self.origin:np.ndarray = origin
        self.orientation:np.ndarray = orientation

        self.R_wp:wp.mat33f = wp_Rzyx(float(orientation[0]), float(orientation[1]), float(orientation[2]))
        self.boresight:wp.vec3f = wp.mul(self.R_wp, wp.vec3(1.0, 0.0, 0.0))

        #RHCP polarization in local frame with propagation on +x = [0, 1, -j]
        self.pol_re = wp.mul(self.R_wp, wp.vec3(0.0, 1.0, 0.0))
        self.pol_im = wp.mul(self.R_wp, wp.vec3(0.0, 0.0, -1.0))

class Receiver:

    def __init__(self, device, origin=(0,0,10), rx_size=(0.035, 0.035, 0.035), n_subreceivers=12, orientation=np.array([0,np.deg2rad(10),0])):

        self.device = device
        self.origin = origin
        self.rx_size = rx_size
        self.n_subreceivers = n_subreceivers
        self.orientation = orientation

        self.R_np:np.ndarray= Rzyx(*orientation)
        self.R_wp:wp.mat33f = wp_Rzyx(float(orientation[0]), float(orientation[1]), float(orientation[2]))
        self.boresight:wp.vec3f = wp.mul(self.R_wp, wp.vec3(1.0, 0.0, 0.0))

        #RHCP polarization in local frame with propagation on +x = [0, 1, -j]
        self.pol_re = wp.mul(self.R_wp, wp.vec3(0.0, 1.0, 0.0))
        self.pol_im = wp.mul(self.R_wp, wp.vec3(0.0, 0.0, -1.0))

        self.mesh, self.verts, self.faces, self.sub_centers, self.face_to_idx = self.create_mesh()

        #self.visualize_mesh()

    def build_ant_pos(self):
        yz = np.array([
            [ 0.00, -0.10],
            [ 0.00, -0.15],
            [-0.05, -0.15],
            [-0.10, -0.15],
            [-0.15, -0.15],
            [-0.15, -0.10],
            [-0.15, -0.05],
            [-0.15,  0.00],
            [-0.10,  0.00],
            [-0.05,  0.00],
            [ 0.00,  0.00],
            [ 0.00, -0.05],
        ], dtype=float)

        # Preserve indexing. Element 10 is [0, 0, 0].
        ant_pos = np.column_stack([
            np.zeros(len(yz)),
            yz[:, 0],
            yz[:, 1],
        ])

        return ant_pos.T

    def get_wp_vec3_origin(self):
        return wp.vec3(float(self.origin[0]), float(self.origin[1]), float(self.origin[2]))

    def visualize_mesh(self):
        verts = self.verts
        idx = self.faces

        faces = [[verts[i] for i in tri] for tri in idx]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        mesh = Poly3DCollection(faces, alpha=0.5)
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)

        # ---- plot vertices ----
        ax.scatter(verts[:,0], verts[:,1], verts[:,2])

        # label vertices
        for i, v in enumerate(verts):
            continue
            ax.text(v[0], v[1], v[2], str(i))

        # label faces
        f2i = np.repeat(np.arange(12), 2)
        for i, f in enumerate(faces):
            v = faces[i][1]
            ax.text(v[0], v[1], v[2], f2i[i])


        # ---- equal aspect ratio ----
        max_range = (verts.max(axis=0) - verts.min(axis=0)).max() / 2.0
        mid = verts.mean(axis=0)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        az = self.orientation[2]

        origin = np.asarray(self.origin, dtype=float)

        arrow_len = 0.12
        dx = arrow_len * np.cos(az)
        dy = arrow_len * np.sin(az)
        dz = 0.0

        ax.quiver(
            origin[0], origin[1], origin[2],
            dx, dy, dz,
            linewidth=3,
            arrow_length_ratio=0.2,
            label="azimuth / +x direction"
        )

        plt.show()
        return None
  
    def create_mesh(self):
        origin = np.asarray(self.origin, dtype=float)

        sx, sy, sz = self.rx_size
        hy = sy * 0.5
        hz = sz * 0.5

        centers_local = self.build_ant_pos().T

        verts_list = []
        idx_list = []

        for k, c in enumerate(centers_local):
            # Patch lies in local YZ plane.
            # Normal is local +X.
            v_local = np.array([
                [c[0], c[1] - hy, c[2] - hz],
                [c[0], c[1] + hy, c[2] - hz],
                [c[0], c[1] + hy, c[2] + hz],
                [c[0], c[1] - hy, c[2] + hz],
            ], dtype=float)

            base = 4 * k

            # Winding chosen for +X normal.
            faces = np.array([
                [base, base + 1, base + 2],
                [base, base + 2, base + 3],
            ], dtype=np.int32)

            verts_list.append(v_local)
            idx_list.append(faces)

        verts_local = np.vstack(verts_list)
        idx = np.vstack(idx_list)

        verts = verts_local @ self.R_np.T + origin
        sub_centers = centers_local @ self.R_np.T + origin

        d = verts - origin
        self.radius_squared = np.max(np.sum(d * d, axis=1))

        d_verts = wp.array(verts, dtype=wp.vec3, device=self.device)
        d_idx = wp.array(idx.reshape(-1), dtype=wp.int32, device=self.device)

        face_to_idx_np = np.repeat(np.arange(12, dtype=np.uint8), 2)

        face_to_idx = wp.array(
            face_to_idx_np,
            dtype=wp.uint8,
            device=self.device
        )

        return (
            wp.Mesh(points=d_verts, indices=d_idx),
            verts,
            idx,
            sub_centers,
            face_to_idx
        )