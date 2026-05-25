import warp as wp
from dataclasses import dataclass

from antennas import *
from raytracer.terrain_generation import *
from antenna_pattern import *

wp.config.quiet = True

# Hit types
HIT_MISS    = wp.uint8(0)
HIT_TERRAIN = wp.uint8(1)
HIT_RX      = wp.uint8(2)

# RX tags
RX_TAG_NONE        = wp.uint8(0)
RX_TAG_NONRECEIVE  = wp.uint8(15)
RX_TAG_TERRAIN_END = wp.uint8(14)

# ---- Raytracer Inputs ----
@wp.struct
class RaytracerConstantsWP:
    max_bounces: int
    max_t: float
    eps: float

    n_rays_mp: int
    n_rays_per_rx: int
    n_rays_per_cell: int

    mp_threshold_dist: float

# CPU side constant and parameters
@dataclass
class RaytracerConstants:
    device: str = "cuda:0" if wp.is_cuda_available() else "cpu"
    max_bounces: int = 2
    max_t: float = 1e4              # Max ray flight time before termination
    eps: float = 1e-4               # Moves the origin of an reflection a tiny distance in direction of the ray to avoid self-hit
    n_rays_mp: int = 1000           # Total number of rays used to hit the terrain
    n_rays_per_rx: int = 1000       # Per RX number of rays for LOS
    n_rays_per_cell: int = 16       # Per terrain cell number of exploration rays (only for presampling.)
    mp_threshold_dist: float = 0.5  # Acceptance distance from rx at closest raypath point for terrain cells. 

    def to_warp(self):
        host = RaytracerConstantsWP()
        host.n_rays_mp = self.n_rays_mp
        host.max_bounces = self.max_bounces
        host.max_t = self.max_t
        host.eps = self.eps
        host.n_rays_per_rx = self.n_rays_per_rx
        host.n_rays_per_cell = self.n_rays_per_cell
        host.mp_threshold_dist = self.mp_threshold_dist
        return host
    

@wp.struct
class RaysInitWP:
    origin: wp.vec3
    dirs: wp.array(dtype=wp.vec3)
    amps: wp.array(dtype=wp.float32)
    tx_boresight: wp.vec3

class RayInit:
    def __init__(self, origin, dirs, amps, tx_boresight) -> None:
        self.origin: wp.vec3 = origin
        self.dirs: wp.array(dtype=wp.vec3) = dirs
        self.tx_boresight = tx_boresight
        self.amps: wp.array(dtype=wp.float32) = amps # Let the electric fields be given in mV/m
        self.n_rays: int = len(dirs)

        self.antenna_pattern_scaling()

    def antenna_pattern_scaling(self):
        wp.launch(
            kernel=calculate_gains,
            dim=self.n_rays,
            inputs= [self.dirs, self.tx_boresight, self.amps],
            device=self.dirs.device,
        )

    def to_warp(self):
        host = RaysInitWP()
        host.origin = self.origin
        host.dirs = self.dirs
        host.amps = self.amps
        return host
    


# ---- Raytracer Outputs ----
@wp.struct
class RaytracerOutputsWP:

    # Complex voltage
    rx_v: wp.array(dtype=wp.float32)
    rx_v_no_mp: wp.array(dtype=wp.float32)
    
    # Complex 3D electric field
    rx_efield: wp.array(dtype=wp.float32)
    rx_efield_no_mp: wp.array(dtype=wp.float32)

@dataclass
class RaytracerOutputs:
    rx_v: wp.array
    rx_v_no_mp: wp.array
    rx_efield: wp.array
    rx_efield_no_mp: wp.array

    @classmethod
    def allocate(cls, device: str, n_rx: int) -> "RaytracerOutputs":
        return cls(
            rx_v=wp.zeros((n_rx, 2), dtype=wp.float32, device=device),
            rx_v_no_mp = wp.zeros((n_rx, 2), dtype=wp.float32, device=device),
            rx_efield=wp.zeros((n_rx, 6), dtype=wp.float32, device=device),
            rx_efield_no_mp=wp.zeros((n_rx, 6), dtype=wp.float32, device=device),
        )

    def to_wp(self) -> RaytracerOutputsWP:
        host = RaytracerOutputsWP()
        host.rx_efield = self.rx_efield
        host.rx_efield_no_mp = self.rx_efield_no_mp
        host.rx_v = self.rx_v
        host.rx_v_no_mp = self.rx_v_no_mp
        return host

    def to_numpy(self):
        return {
            "rx_efield": self.rx_efield.numpy(),
            "rx_efield_no_mp": self.rx_efield_no_mp.numpy(),
            "rx_v": self.rx_v.numpy(),
            "rx_v_no_mp": self.rx_v_no_mp.numpy()
        }

@wp.struct
class RaytracerDebugsWP:
    end_hits: wp.array(dtype=wp.uint8) # 0=Miss, 1-11=RX element, 15=Terrain
    path_lengths: wp.array(dtype=float) # Length of each ray's path.
    terrain_hitcounts: wp.array(dtype=int, ndim=2) # 2D grid of terrain holding count of the reflections.

@dataclass
class RaytracerDebugs:
    end_hits: wp.array
    path_lengths: wp.array
    terrain_hitcounts: wp.array

    @classmethod
    def allocate(cls, device, n_rays: int, Nx:int, Ny:int) -> "RaytracerDebugs":
        return cls(
            end_hits=wp.zeros(n_rays, dtype=wp.uint8, device=device),
            path_lengths=wp.zeros(n_rays, dtype=wp.float32, device=device),
            terrain_hitcounts = wp.zeros((Ny, Nx), dtype=wp.int32, device=device)
        )
    
    def to_wp(self):
        host = RaytracerDebugsWP()
        host.end_hits = self.end_hits
        host.path_lengths = self.path_lengths
        host.terrain_hitcounts = self.terrain_hitcounts
        return host

    def to_numpy(self):
        return dict(
            path_lengths = self.path_lengths.numpy(),
            end_hits = self.end_hits.numpy(),
            terrain_hitcounts = self.terrain_hitcounts.numpy(),
        )
    
# Record per ray hit data
@wp.struct
class HitRecord:
    type: wp.uint8 # 0=none, 1=terrain, 2=rx
    t: float
    hit_point: wp.vec3
    normal: wp.vec3
    rx_tag: wp.uint8   # 0=not rx, 1...14=sub rx i, 15=non_receiving
    min_rx_dist_sqrd: wp.float32