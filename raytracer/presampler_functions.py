import warp as wp
import numpy as np
import matplotlib.pyplot as plt

from raytracer.utils_rt import *
from raytracer.dataclasses import *
from raytracer.scene import *

@wp.kernel
def uniform_k_cells_dir_sampling(
    const: RaytracerConstantsWP,
    scene: SceneWP,
    out_dir_tx: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    k = const.n_rays_per_cell
    seed = 0

    nx = scene.Nx
    ny = scene.Ny
    dx = scene.dx
    dy = scene.dy

    ncx = nx - 1
    ncy = ny - 1
    num_cells = ncx * ncy

    cell_id = tid // k
    if cell_id >= num_cells:
        return

    j = cell_id // ncx
    i = cell_id - j * ncx


    x0 = scene.terrain_origin[0]
    y0 = scene.terrain_origin[1]
    x0c = x0 + wp.float32(i) * dx
    y0c = y0 + wp.float32(j) * dy

    r = wp.rand_init(seed, tid)
    u = wp.randf(r)
    v = wp.randf(r)

    px = x0c + u * dx
    py = y0c + v * dy

    pz = bilerp_z(px, py, scene)
    p = wp.vec3(px, py, pz)

    dtx = p - scene.tx_origin
    out_dir_tx[tid] = wp.normalize(dtx)

@wp.kernel
def terrain_presampler_kernel(
    const: RaytracerConstantsWP,
    scene: SceneWP,
    dirs: wp.array(dtype=wp.vec3),
    dir_scores: wp.array(dtype=wp.float32)
):
    
    tid = wp.tid()
    if tid >= dirs.shape[0]:
        return
    
    target = scene.rx_origin
    best = wp.float32(1.0e30)
    origin = scene.tx_origin
    dir = dirs[tid]

    for b in range(5): # Presampling is decently cheap...

        min_rx_dist, min_rx_t = min_ray_distance(origin, dir, target)
        q = wp.mesh_query_ray(scene.terrain_mesh_id, origin, dir, const.max_t)

        if b != 0:
            cand = wp.dot(min_rx_dist, min_rx_dist)
            if min_rx_t < 0.0:
                v = target - origin
                cand = wp.dot(v, v)
            best = wp.min(best, cand)

        if not q.result:
            break

        hit_point = origin + q.t * dir
        normal = scene.normals[q.face]
        reflection = wp.normalize(dir - 2.0 * wp.dot(dir, normal) * normal)

        origin = hit_point + const.eps * reflection
        dir = reflection

    dir_scores[tid] = best

@wp.kernel
def reduce_score_per_cell(scores: wp.array(dtype=wp.float32),
                         best: wp.array(dtype=wp.float32),
                         k: wp.int32):
    cell = wp.tid()

    m = wp.float32(1.0e30)
    base = cell * k

    for r in range(k):
        s = scores[base + r]
        m = wp.min(m, s)

    best[cell] = m

@wp.kernel
def build_accepted_cells(best: wp.array(dtype=wp.float32),
                         accepted_ids: wp.array(dtype=wp.int32),
                         accepted_count: wp.array(dtype=wp.int32),
                         threshold_sqrd: wp.float32):
    cell = wp.tid()

    if best[cell] < threshold_sqrd:
        idx = wp.atomic_add(accepted_count, 0, 1)
        accepted_ids[idx] = cell

@wp.kernel
def solid_angles_of_accepted_cells(
    scene:SceneWP, 
    accepted_ids:wp.array(dtype=wp.int32),
    solid_angles_mp:wp.array(dtype=wp.float32),
    solid_angles_rx:wp.array(dtype=wp.float32),
    ):

    tid = wp.tid()

    #Cells solid angles:
    if tid >= scene.rx_n_rx:
        idx = tid - scene.rx_n_rx
        cell_id = accepted_ids[idx]

        nx = scene.Nx
        ny = scene.Ny
        dx = scene.dx
        dy = scene.dy

        ncx = nx - 1
        ncy = ny - 1

        j = cell_id // ncx
        i = cell_id - j * ncx

        z0 = scene.terrain_heightmap[j,i]
        z1 = scene.terrain_heightmap[j,i+1]
        z2 = scene.terrain_heightmap[j+1,i]
        z3 = scene.terrain_heightmap[j+1,i+1]
        
        x0 = scene.terrain_origin[0] + wp.float32(i) * dx
        x1 = scene.terrain_origin[0] + wp.float32(i+1) * dx

        y0 = scene.terrain_origin[1] + wp.float32(j) * dy
        y1 = scene.terrain_origin[1] + wp.float32(j+1) * dy

        v0 = wp.vec3(x0, y0, z0)
        v1 = wp.vec3(x1, y0, z1)
        v2 = wp.vec3(x0, y1, z2)
        v3 = wp.vec3(x1, y1, z3)

        solid_angles_mp[idx] = quad_solid_angle(v0, v1, v2, v3, scene.tx_origin)

    # Receiver solid angles:
    else:

        base = wp.int(4*tid)
        rv0 = scene.rx_verts[base]
        rv1 = scene.rx_verts[base + wp.int(1)]
        rv2 = scene.rx_verts[base + wp.int(2)]
        rv3 = scene.rx_verts[base + wp.int(3)]

        solid_angles_rx[tid] = quad_solid_angle(rv0,rv1,rv2,rv3, scene.tx_origin)

@wp.func
def tri_solid_angle_from_origin(a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    #https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron
    la = wp.length(a)
    lb = wp.length(b)
    lc = wp.length(c)

    num = wp.abs(wp.dot(a, wp.cross(b, c)))
    den = la*lb*lc + wp.dot(a, b)*lc + wp.dot(b, c)*la + wp.dot(c, a)*lb
    return wp.float32(2.0) * wp.atan2(num, den)

@wp.func
def quad_solid_angle(v0: wp.vec3, v1: wp.vec3, v2: wp.vec3, v3: wp.vec3, origin: wp.vec3) -> wp.float32:
    #https://en.wikipedia.org/wiki/Solid_angle#Tetrahedron

    a0 = v0 - origin
    a1 = v1 - origin
    a2 = v2 - origin
    a3 = v3 - origin

    return tri_solid_angle_from_origin(a0, a1, a3) + tri_solid_angle_from_origin(a0, a3, a2)


@wp.kernel
def sample_dirs_from_accepted_cells(
    const: RaytracerConstantsWP,
    scene: SceneWP,
    accepted_ids: wp.array(dtype=wp.int32),
    accepted_count: wp.array(dtype=wp.int32),
    solid_angles_rx: wp.array(dtype=wp.float32),
    solid_angles_mp: wp.array(dtype=wp.float32),
    out_pts: wp.array(dtype=wp.vec3),
    out_dirs: wp.array(dtype=wp.vec3),
    out_weights: wp.array(dtype=wp.float32),
    
):
    tid = wp.tid()
    seed = 0
    r = wp.rand_init(seed, tid)

    # Terrain/Multipath sampling
    if tid >= const.n_rays_per_rx * scene.rx_n_rx:

        nx = scene.Nx
        ny = scene.Ny
        dx = scene.dx
        dy = scene.dy
        ncx = nx - 1
        ncy = ny - 1

        m = accepted_count[0]

        # Sample uniformely among the accepted cells.
        u = wp.randf(r)
        aidx = wp.clamp(wp.int32(u * wp.float32(m)), 0, m - 1)
        cell_id = accepted_ids[aidx]

        # map cell_id -> (i, j)
        j = cell_id // ncx
        i = cell_id - j * ncx
        
        x0 = scene.terrain_origin[0]
        y0 = scene.terrain_origin[1]
        x = x0 + wp.float32(i) * dx
        y = y0 + wp.float32(j) * dy

        #Random offset from cell lower left corner.
        eps = wp.float32(1e-4) # eps avoids drifting to wrong cells due to numerical error.
        tx = eps + (wp.float32(1.0) - wp.float32(2.0)*eps) * wp.randf(r)
        ty = eps + (wp.float32(1.0) - wp.float32(2.0)*eps) * wp.randf(r)
        px = x + tx * dx
        py = y + ty * dy
        pz = bilerp_z(px, py, scene)

        p = wp.vec3(px, py, pz)
        out_pts[tid] = p
        out_dirs[tid] = wp.normalize(p - scene.tx_origin)
        out_weights[tid] = wp.float32(m) * solid_angles_mp[aidx] / wp.float32(const.n_rays_mp)
        return

    # Rx sampling
    else:
        rx_idx = wp.min(tid // const.n_rays_per_rx, scene.rx_n_rx - 1)
        v_idx = rx_idx * 4

        v0 = scene.rx_verts[v_idx]
        v1 = scene.rx_verts[v_idx+1]
        v3 = scene.rx_verts[v_idx+3]

        u = wp.randf(r)
        v = wp.randf(r)
        
        #Bilinear sampling
        pt =  v0 + u*(v1-v0) + v*(v3-v0)

        out_pts[tid] = pt
        out_dirs[tid] = wp.normalize(pt - scene.tx_origin)
        out_weights[tid] = solid_angles_rx[rx_idx] / wp.float32(const.n_rays_per_rx)
        return



def plot_presampler_scores(scores):

    print(f"Total rays: {len(scores)}")
    print(f"Finite scores: {len(scores)}")
    print(f"Min: {scores.min():.4e}")
    print(f"Max: {scores.max():.4e}")
    print(f"Mean: {scores.mean():.4e}")

    plt.figure(figsize=(8, 4))
    plt.hist(scores, bins=200)
    plt.yscale("log")
    plt.title("Presampler Distance^2 Histogram")
    plt.xlabel("Distance^2")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.show()

    sorted_scores = np.sort(scores)
    plt.figure(figsize=(8, 4))
    plt.plot(sorted_scores)
    plt.yscale("log")
    plt.title("Sorted Distance^2")
    plt.xlabel("Ray index (sorted)")
    plt.ylabel("Distance^2 (log)")
    plt.tight_layout()
    plt.show()

def plot_cell_heatmap(best_cell: np.ndarray, Nx: int, Ny: int,
                      title="Most promising cells heatmap",
                      log_scale=True,
                      cbar_label=None):
    """
    best_cell: flat ((Ny-1)*(Nx-1),) or (Ny-1, Nx-1)
    """

    plot_data = best_cell.reshape((Ny-1, Nx-1))

    if log_scale:
        plot_data = np.log1p(plot_data)

    plt.figure(figsize=(12, 3))
    plt.imshow(plot_data, origin="lower", aspect="auto")
    plt.colorbar(label=(cbar_label or ("log1p(score)" if log_scale else "score")))
    plt.title(title)
    plt.xlabel("x cell index")
    plt.ylabel("y cell index")
    plt.tight_layout()
    plt.show()