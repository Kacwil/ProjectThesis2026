import warp as wp
from .physics import *
from raytracer.complex_math import *

@wp.func
def fresnel_reflect_lossy(
    efield: EField,
    ray_dir: wp.vec3,
    normal: wp.vec3,
):

    # Soil parameters
    eps_r = wp.float32(10.0)
    sigma = wp.float32(0.2)

    eps0 = wp.float32(8.854187817e-12)
    pi = wp.float32(3.14159265358979323846)
    omega = wp.float32(2.0) * pi * efield.freq

    # complex relative permittivity: eps_r - j*sigma/(omega*eps0)
    epsc_re = eps_r
    epsc_im = -sigma / (omega * eps0)

    k_i = wp.normalize(ray_dir)
    n = wp.normalize(normal)

    # force normal to oppose incoming ray
    if wp.dot(k_i, n) > 0.0:
        n = -n

    # plane of incidence basis
    s = wp.cross(k_i, n)
    if wp.length(s) < wp.float32(1.0e-6):
        ref = wp.vec3(1.0, 0.0, 0.0)
        s = ref - k_i * wp.dot(ref, k_i)
        if wp.length(s) < wp.float32(1.0e-6):
            ref = wp.vec3(0.0, 1.0, 0.0)
            s = ref - k_i * wp.dot(ref, k_i)

    s = wp.normalize(s)
    p_i = wp.normalize(wp.cross(s, k_i))

    Es_re = wp.dot(efield.re, s)
    Es_im = wp.dot(efield.im, s)
    Ep_re = wp.dot(efield.re, p_i)
    Ep_im = wp.dot(efield.im, p_i)

    cos_theta_i = wp.clamp(-wp.dot(k_i, n), 0.0, 1.0)
    sin2_theta_i = wp.float32(1.0) - cos_theta_i * cos_theta_i

    g_re, g_im = c_sqrt(epsc_re - sin2_theta_i, epsc_im)

    rs_num_re, rs_num_im = c_sub(cos_theta_i, 0.0, g_re, g_im)
    rs_den_re, rs_den_im = c_add(cos_theta_i, 0.0, g_re, g_im)
    rs_re, rs_im = c_div(rs_num_re, rs_num_im, rs_den_re, rs_den_im)

    ec_re, ec_im = c_mul(epsc_re, epsc_im, cos_theta_i, 0.0)
    rp_num_re, rp_num_im = c_sub(ec_re, ec_im, g_re, g_im)
    rp_den_re, rp_den_im = c_add(ec_re, ec_im, g_re, g_im)
    rp_re, rp_im = c_div(rp_num_re, rp_num_im, rp_den_re, rp_den_im)

    # reflected direction
    k_r = wp.normalize(k_i - wp.float32(2.0) * wp.dot(k_i, n) * n)
    p_r = wp.normalize(wp.cross(s, k_r))

    # apply complex reflection coefficients
    Esr_re, Esr_im = c_mul(rs_re, rs_im, Es_re, Es_im)
    Epr_re, Epr_im = c_mul(rp_re, rp_im, Ep_re, Ep_im)

    # rebuild reflected vector field
    E_re = Esr_re * s + Epr_re * p_r
    E_im = Esr_im * s + Epr_im * p_r

    return construct_efield(E_re, E_im, efield.freq), k_r