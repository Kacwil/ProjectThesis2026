# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportReturnType=false
# pyright: reportInvalidTypeForm=false
# pyright: reportIndexIssue=false

import warp as wp
from antennas import *
from raytracer.terrain_generation import *

# GPU side object
@wp.struct
class SceneWP:
    freq: float

    Nx: int
    Ny: int
    dx: float
    dy: float
    terrain_origin: wp.vec2
    terrain_mesh_id: wp.uint64
    terrain_heightmap: wp.array2d(dtype=wp.float32)
    normals: wp.array(dtype=wp.vec3)

    rx_mesh_id: wp.uint64
    rx_n_rx: wp.int32
    rx_origin: wp.vec3
    rx_boresight: wp.vec3
    rx_R: wp.mat33f
    rx_pol_re: wp.vec3
    rx_pol_im: wp.vec3
    rx_radius_sqrd: float
    rx_verts:wp.array(dtype=wp.vec3)
    rx_face_to_idx:wp.array(dtype=wp.uint8)

    tx_origin: wp.vec3
    tx_boresight: wp.vec3
    tx_R: wp.mat33f
    tx_pol_re: wp.vec3
    tx_pol_im: wp.vec3

# CPU side object
class Scene:
    def __init__(self, tx, rx, terrain, freq = 2.48e9):
        self.tx:Transmitter = tx
        self.rx:Receiver = rx
        self.terrain:UniformTerrain3D = terrain 
        self.freq = freq

    def to_warp(self, device):
        host = SceneWP()

        # Terrain
        host.freq = self.freq
        
        host.Nx = self.terrain.Nx
        host.Ny = self.terrain.Ny
        host.dx = self.terrain.dx
        host.dy = self.terrain.dy
        host.terrain_origin = wp.vec2(self.terrain.xs[0], self.terrain.ys[0])
        host.terrain_mesh_id = wp.uint64(self.terrain.mesh.id)
        host.terrain_heightmap = wp.array2d(self.terrain.zs, device=device)
        host.normals = self.terrain.face_normals

        # Receiver
        host.rx_mesh_id = wp.uint64(self.rx.mesh.id)
        host.rx_n_rx = wp.int32(self.rx.n_subreceivers)
        host.rx_origin = self.rx.get_wp_vec3_origin()
        host.rx_boresight = self.rx.boresight
        host.rx_R = self.rx.R_wp
        host.rx_radius_sqrd = self.rx.radius_squared
        host.rx_verts = wp.array(self.rx.verts, dtype=wp.vec3, device=device)
        host.rx_pol_re = self.rx.pol_re
        host.rx_pol_im = self.rx.pol_im
        host.rx_face_to_idx = self.rx.face_to_idx

        # Transmitter
        host.tx_origin = self.tx.origin
        host.tx_boresight = self.tx.boresight
        host.tx_R = self.tx.R_wp
        host.tx_pol_re = self.tx.pol_re
        host.tx_pol_im = self.tx.pol_im

        return host


# --- Utility functions ---

@wp.func
def world_xy_to_2D_index(x: wp.float32, y: wp.float32, scene: SceneWP) -> wp.vec2i:
    # floats x,y -> integer cell index (ix, iy)
    x0 = scene.terrain_origin[0]
    y0 = scene.terrain_origin[1]
    dx = scene.dx
    dy = scene.dy
    Nx = scene.Nx
    Ny = scene.Ny

    ix = wp.int32(wp.floor((x - x0) / dx))
    iy = wp.int32(wp.floor((y - y0) / dy))

    ix = wp.clamp(ix, 0, Nx - 2)
    iy = wp.clamp(iy, 0, Ny - 2)

    return wp.vec2i(ix, iy)

@wp.func
def bilerp_z(x: float, y: float, scene:SceneWP) -> float:

    """
    Find height z for a pair of floats x,y by using interpolation.
    """

    x0 = scene.terrain_origin[0]
    y0 = scene.terrain_origin[1]
    dx = scene.dx
    dy = scene.dy

    zs = scene.terrain_heightmap

    index = world_xy_to_2D_index(x,y, scene)
    ix = index[0]
    iy = index[1]

    fx = (x - x0) / dx
    fy = (y - y0) / dy

    tx = wp.clamp(fx - float(ix), 0.0, 1.0)
    ty = wp.clamp(fy - float(iy), 0.0, 1.0)

    z00 = zs[iy, ix]
    z10 = zs[iy, ix + 1]
    z01 = zs[iy + 1, ix]
    z11 = zs[iy + 1, ix + 1]

    z0 = z00 * (1.0 - tx) + z10 * tx
    z1 = z01 * (1.0 - tx) + z11 * tx
    return z0 * (1.0 - ty) + z1 * ty