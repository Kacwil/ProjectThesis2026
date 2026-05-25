# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportReturnType=false
# pyright: reportIndexIssue=false

import warp as wp

from utils import *
from .physics import *
from .dataclasses import *
from .utils_rt import *
from .presampler import terrain_presampler
from raytracer.terrain_generation import *
from plotting import *
from raytracer.scene import *
from antenna_pattern import *

from raytracer.fresnel import fresnel_reflect_lossy

@wp.kernel
def raycast_kernel(
    const: RaytracerConstantsWP,
    scene: SceneWP,
    rays: RaysInitWP,
    outs: RaytracerOutputsWP,
    debug: RaytracerDebugsWP,
):
    
    #--- Inits ---
    tid = wp.tid()
    if tid >= rays.dirs.shape[0]:
        return
    origin = scene.tx_origin
    dir = rays.dirs[tid]
    efield = init_E_fields(dir, rays.amps[tid], scene.freq, scene.tx_pol_re, scene.tx_pol_im)
    LOS = (tid < (const.n_rays_per_rx * scene.rx_n_rx))

    for bounce in range(const.max_bounces):

        rec = trace_one_segment(const, scene, debug, origin, dir)

        # Miss
        if (rec.type == HIT_MISS or rec.rx_tag == RX_TAG_NONRECEIVE):
            break

        efield = propagate(efield, rec.t)

        # Receiver hit
        if rec.type == HIT_RX:
            handle_rx_hit(rec, dir, efield, scene, outs, debug, bounce, tid, LOS, scene.rx_boresight)
            break

        # Terrain hit
        origin, dir, efield = handle_terrain_hit(rec, efield, const, debug, dir, tid)

@wp.func
def trace_one_segment(
    const: RaytracerConstantsWP,
    scene: SceneWP,
    debug: RaytracerDebugsWP,
    origin: wp.vec3,
    dir: wp.vec3,
    ):

    rec = init_record()
    q = wp.mesh_query_ray(scene.terrain_mesh_id, origin, dir, const.max_t)
    q_rx = wp.mesh_query_ray(scene.rx_mesh_id, origin, dir, const.max_t) 

    # We hit the receiver first before the terrain:
    if q_rx.result and (not q.result or q_rx.t < q.t):
        rec.type = wp.uint8(2)
        rec.t = q_rx.t
        rec.hit_point = origin + q_rx.t * dir

        # Non-receiving surface is hit:
        if q_rx.face > scene.rx_n_rx * 2:
            rec.rx_tag = wp.uint8(15)

        # Receiving surface is hit:
        else:
            rec.rx_tag = scene.rx_face_to_idx[q_rx.face]

        # For debugging:
        if q.result:
            rec.hit_point = origin + q.t * dir
            idx = world_xy_to_2D_index(rec.hit_point[0], rec.hit_point[1], scene)
            wp.atomic_add(debug.terrain_hitcounts, idx[1], idx[0], 1)

        return rec

    # We didn't hit the receiver first:
    if not (q.result):
        return rec

    rec.type = wp.uint8(1)
    rec.t = q.t
    rec.hit_point = origin + q.t * dir
    rec.normal = scene.normals[q.face]

    idx = world_xy_to_2D_index(rec.hit_point[0], rec.hit_point[1], scene)
    wp.atomic_add(debug.terrain_hitcounts, idx[1], idx[0], 1)
    return rec

@wp.func
def init_record():
    rec = HitRecord()
    rec.type = wp.uint8(0)
    rec.rx_tag = wp.uint8(0)
    rec.t = 0.0
    rec.hit_point = wp.vec3(0.0,0.0,0.0)
    rec.normal = wp.vec3(0.0,0.0,0.0)
    return rec

@wp.func
def handle_rx_hit(rec:HitRecord, direction:wp.vec3, efield:EField, scene:SceneWP, outs:RaytracerOutputsWP, debug:RaytracerDebugsWP, bounce:int, tid:int, LOS:bool, boresight:wp.vec3):

    tag_idx = rec.rx_tag

    cos_rel_angle = wp.dot(-direction, boresight)
    cos_rel_angle = wp.clamp(cos_rel_angle, -1.0, 1.0)
    rel_angle = wp.acos(cos_rel_angle)
    gain = wp.sqrt(calculate_gain(rel_angle))

    efield.re = gain * efield.re
    efield.im = gain * efield.im

    debug.end_hits[tid] = rec.rx_tag
    debug.path_lengths[tid] += rec.t

    efield = pathloss_attenuation(efield, debug.path_lengths[tid])

    if bounce == 0:
        # Exclude multipath

        # Ignore MP rays incident to LOS / avoid double counting.
        if LOS:
            sum_efield_at_rx(efield, outs.rx_efield_no_mp, tag_idx)
            sum_complex_voltage_at_rx(efield, direction, outs.rx_v_no_mp, tag_idx, scene.rx_pol_re, scene.rx_pol_im)
        else:
            return
        
    # Include Multipath
    sum_efield_at_rx(efield, outs.rx_efield, tag_idx)
    sum_complex_voltage_at_rx(efield, direction, outs.rx_v, tag_idx, scene.rx_pol_re, scene.rx_pol_im)

    return

@wp.func
def handle_terrain_hit(rec:HitRecord, efield:EField, const:RaytracerConstantsWP, debug:RaytracerDebugsWP, dir:wp.vec3, tid:int):

    #efield = simple_reflect(efield)
    #reflection = wp.normalize(dir - 2.0 * wp.dot(dir, rec.normal) * rec.normal)

    #efield, reflection = fresnel_reflect(efield, dir, rec.normal)

    efield, reflection = fresnel_reflect_lossy(efield, dir, rec.normal)

    origin = rec.hit_point + const.eps * reflection
    dir = reflection

    debug.end_hits[tid] = RX_TAG_TERRAIN_END
    debug.path_lengths[tid] += rec.t

    return origin, dir, efield


def raycast(scene:Scene, const:RaytracerConstants):

    n_los = const.n_rays_per_rx * scene.rx.n_subreceivers
    n_mp = const.n_rays_mp

    const_wp = const.to_warp()
    scene_wp = scene.to_warp(const.device)
    
    outs = RaytracerOutputs.allocate(const.device, scene.rx.n_subreceivers)
    outs_wp = outs.to_wp()

    debug = RaytracerDebugs.allocate(const.device, (n_los+n_mp), scene.terrain.Nx, scene.terrain.Ny)
    debug_wp = debug.to_wp()

    pts, dirs, weights = terrain_presampler(const_wp, scene_wp)
    rays = RayInit(scene.tx.origin, dirs, weights, scene.tx.boresight)
    rays_wp = rays.to_warp()
    
    wp.launch(
        kernel=raycast_kernel,
        dim=(n_los+n_mp),
        inputs= [const_wp, scene_wp, rays_wp, outs_wp, debug_wp],
        device=const.device,
    )

    return outs, debug
