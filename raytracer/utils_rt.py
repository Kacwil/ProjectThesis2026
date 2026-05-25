# pyright: reportGeneralTypeIssues=false
# pyright: reportOperatorIssue=false
# pyright: reportAttributeAccessIssue=false
# pyright: reportCallIssue=false
# pyright: reportArgumentType=false
# pyright: reportReturnType=false

import warp as wp
    
@wp.func
def min_ray_distance(ray_origin:wp.vec3, ray_dir:wp.vec3, target:wp.vec3):
    """
    Computest the point along the ray which is closest to a target.
    """
    ao = ray_origin - target
    t_min = -wp.dot(ao, ray_dir)
    p_min = ao + ray_dir * t_min
    return p_min, t_min

@wp.func
def wp_Rzyx(xa:float, ya:float, za:float) -> wp.mat33f:

    cx, sx = wp.cos(xa), wp.sin(xa)
    cy, sy = wp.cos(ya), wp.sin(ya)
    cz, sz = wp.cos(za), wp.sin(za)

    Rx = wp.mat33(
        1.0, 0.0, 0.0,
        0.0, cx, -sx,
        0.0, sx, cx
    )

    Ry = wp.mat33(
        cy, 0.0, sy,
        0.0, 1.0, 0.0,
        -sy, 0.0, cy
    )

    Rz = wp.mat33(
        cz, -sz, 0.0,
        sz, cz, 0.0,
        0.0, 0.0, 1.0
    )

    R = wp.mul(Rz, wp.mul(Ry, Rx))
    return R


@wp.func
def build_transverse_basis(e1:wp.vec3, reference:wp.vec3 = wp.vec3(1.0, 0.0, 0.0), fallback:wp.vec3 = wp.vec3(0.0, 1.0, 0.0)):

    """
    Create a orthogonal basis where e2, e3 are transverse to e1.
    """

    e1 = wp.normalize(e1)
    e2 = reference - e1 * wp.dot(reference, e1)

    if wp.length(e2) < wp.float32(1.0e-6):
        e2 = fallback - e1 * wp.dot(fallback, e1)

    e2 = wp.normalize(e2)
    e3 = wp.normalize(wp.cross(e1, e2))

    return e1, e2, e3