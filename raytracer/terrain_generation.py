import numpy as np
import warp as wp
from raytracer.dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class UniformTerrain3D:
    xs: np.ndarray
    ys: np.ndarray
    zs: np.ndarray
    dx: float
    dy: float
    Nx: int
    Ny: int
    mesh: wp.Mesh
    face_normals: wp.array(dtype=wp.vec3)
    verts_np: np.ndarray | None
    idx_np: np.ndarray | None

    @classmethod
    def from_xyz(cls, n_waves: int, steps_per_unit: int, device, random:bool = False, x_size = [-50, 250], y_size=[-10, 10]):
        
        if random:
            freqs  = np.random.uniform(0.01, 0.01, size=n_waves)
            phases = np.random.uniform(0.0, 2*np.pi, size=n_waves)
            amps   = np.random.uniform(0.0, 1.0, size=n_waves) 
            thetas = np.random.uniform(0.0, 2*np.pi, size=n_waves)

        else:
            freqs = np.array([0.12, 0.01, 0.06])
            phases = np.array([-1.2, 0.01, 0.01])
            amps = np.array([0.05, 0.1, 0.1]) * 0
            thetas = np.array([np.deg2rad(80), np.deg2rad(15), np.deg2rad(30)])

        Nx = int(round(abs(x_size[1] - x_size[0]) * steps_per_unit)) + 1
        Ny = int(round(abs(y_size[1] - y_size[0]) * steps_per_unit)) + 1

        xs = np.linspace(x_size[0], x_size[1], Nx)
        ys = np.linspace(y_size[0], y_size[1], Ny)

        X, Y = np.meshgrid(xs, ys, indexing="xy")
        zs = np.zeros((Ny, Nx), dtype=np.float32)

        for A, f, ph, th in zip(amps, freqs, phases, thetas):
            proj = np.cos(th) * X + np.sin(th) * Y
            zs += A * np.sin(2*np.pi * f * proj + ph)

        dx, dy = float(xs[1] - xs[0]), float(ys[1] - ys[0])

        print(np.max(zs))

        mesh, face_normals, verts_np, idx_np = cls.heightmap_to_mesh(xs, ys, zs, device)
        return cls(xs, ys, zs, dx, dy, Nx, Ny, mesh, face_normals, verts_np, idx_np)

    @staticmethod
    def heightmap_to_mesh(xs, ys, zs, device):

        Nx = xs.shape[0]
        Ny = ys.shape[0]

        X, Y = np.meshgrid(xs, ys, indexing="xy")
        vertices_np = np.column_stack([X.ravel(), Y.ravel(), zs.ravel()]).astype(np.float32)

        i = np.arange(Ny - 1)[:, None]
        j = np.arange(Nx - 1)[None, :]

        v00 = i * Nx + j
        v10 = (i + 1) * Nx + j
        v01 = i * Nx + (j + 1)
        v11 = (i + 1) * Nx + (j + 1)

        tri1 = np.stack([v00, v11, v10], axis=-1)
        tri2 = np.stack([v00, v01, v11], axis=-1)

        idx = np.stack([tri1, tri2], axis=2).reshape(-1, 3).astype(np.int32)
        indices_flat = idx.reshape(-1).astype(np.int32)

        vertices = wp.array(vertices_np, dtype=wp.vec3, device=device)
        indices  = wp.array(indices_flat, dtype=wp.int32, device=device)

        v0 = vertices_np[idx[:, 0]]
        v1 = vertices_np[idx[:, 1]]
        v2 = vertices_np[idx[:, 2]]
        n = np.cross(v1 - v0, v2 - v0)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        face_normals = wp.array(n, dtype=wp.vec3, device=device)

        return wp.Mesh(points=vertices, indices=indices), face_normals, vertices_np, idx

    def interp_z(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """
        Get z values for a arrays of floating points px, py by interpolation
        """
        xs, ys, zs = self.xs, self.ys, self.zs
        dx, dy = self.dx, self.dy

        ncx = self.Nx - 1
        ncy = self.Ny - 1

        i = ((px - xs[0]) / dx).astype(int)
        j = ((py - ys[0]) / dy).astype(int)

        i = np.clip(i, 0, ncx - 1)
        j = np.clip(j, 0, ncy - 1)

        x_base = xs[i]
        y_base = ys[j]

        tx = (px - x_base) / dx
        ty = (py - y_base) / dy

        z00 = zs[j,     i]
        z10 = zs[j,     i+1]
        z01 = zs[j+1,   i]
        z11 = zs[j+1,   i+1]

        pz = (
            (1 - tx) * (1 - ty) * z00 +
            tx * (1 - ty) * z10 +
            (1 - tx) * ty * z01 +
            tx * ty * z11
        )

        return pz

    def plot_heightmap(self):

        xs, ys, zs = self.xs, self.ys, self.zs
        extent = [xs[0], xs[-1], ys[0], ys[-1]]

        plt.figure()
        plt.imshow(zs, origin="lower", extent=extent, aspect="auto")
        plt.colorbar(label="z")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Terrain heightmap (z)")
        plt.tight_layout()
        plt.show()

