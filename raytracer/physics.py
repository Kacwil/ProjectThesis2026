import warp as wp
import math

from raytracer.utils_rt import *
from raytracer.complex_math import *

c = 299_792_458.0 # [m/s]
freq = 2.402e9 #TODO: unhardcode this to support multiple channels

@wp.struct
class EField:
    re: wp.vec3
    im: wp.vec3
    freq: float

@wp.func
def construct_efield(Ere:wp.vec3, Eim:wp.vec3, freq:float) -> EField:
    efield = EField()
    efield.re = Ere
    efield.im = Eim
    efield.freq = freq
    return efield

@wp.func
def sum_efields(re1:wp.vec3, re2:wp.vec3, im1:wp.vec3, im2:wp.vec3, freq:float) -> EField:
    return construct_efield(re1+re2, im1+im2, freq)

@wp.func
def normalize(v: wp.vec3) -> wp.vec3:
    n = wp.sqrt(wp.dot(v, v)) + 1e-12
    return v / n

@wp.func
def rotate_efield(efield:EField, phase:float) -> EField:

    re, im = c_vector_rot(efield.re, efield.im, phase)
    return construct_efield(re, im, efield.freq)


@wp.func
def propagate(efield:EField, pathlength:float) -> EField:

    wavelength = c / efield.freq
    wavenumber = (2.0 * math.pi) / wavelength
    phase = -wavenumber * pathlength

    return rotate_efield(efield, phase)

@wp.func
def pathloss_attenuation(efield:EField, total_pathlength:float) -> EField:

    atten = wp.float32(1.0) / (total_pathlength + 1e-6)
    return construct_efield(atten * efield.re, atten * efield.im, efield.freq)


@wp.func
def init_E_fields(direction:wp.vec3, amp:wp.float32, freq:float, tx_pol_re:wp.vec3, tx_pol_im:wp.vec3) -> EField:
    """ 
    Initializes the E fields in 3D.
    A RHCP wave travelling in +z direction has E = [a, -ja ,0] (2.48a)
    """

    direction, e2, e3 = build_transverse_basis(direction, tx_pol_re, tx_pol_im)

    scale = amp * wp.float32(0.7071067811865476) # scale by 1/sqrt(2) for circular polarization

    real = e2 * scale
    imag = -e3 * scale 

    return construct_efield(real, imag, freq)

@wp.func
def simple_reflect(efield:EField) -> EField:

    reflection_coefficient = 0.15

    efield = rotate_efield(efield, wp.radians(90.0))
    re = reflection_coefficient * efield.re
    im = reflection_coefficient * efield.im

    return construct_efield(re, im, efield.freq)

@wp.func
def fresnel_reflect(efield: EField, ray_dir:wp.vec3, normal:wp.vec3):

    n1 = 1.0
    n2 = 3.0
    eta = n1 / n2

    # Basis vectors
    s = wp.normalize(wp.cross(ray_dir, normal))
    p  = wp.normalize(wp.cross(s, ray_dir))

    # Scalar field amplitudes in incident basis
    Es_re = wp.dot(efield.re, s)
    Es_im = wp.dot(efield.im, s)
    Ep_re = wp.dot(efield.re, p)
    Ep_im = wp.dot(efield.im, p)

    cos_theta_i = - wp.clamp(-wp.dot(ray_dir, normal), 0.0, 1.0)

    sin2_theta_i = 1.0 - cos_theta_i * cos_theta_i
    sin2_theta_t = eta * eta * sin2_theta_i

    # Total internal refraction:
    if sin2_theta_t > 1.0:
        rs = 1.0
        rp = 1.0

    # Fresnel:
    else:
        cos_theta_t = wp.sqrt(1.0 - sin2_theta_t)
        rs = (n1*cos_theta_i - n2*cos_theta_t) / (n1*cos_theta_i + n2*cos_theta_t)
        rp = (n2*cos_theta_i - n1*cos_theta_t) / (n2*cos_theta_i + n1*cos_theta_t)

    # Reflected direction
    ref = ray_dir - 2.0 * wp.dot(ray_dir, normal) * normal
    ref = wp.normalize(ref)

    # Reflected p basis
    p_ref = wp.normalize(wp.cross(s, ref))

    # Rebuild reflected field
    E_real = (rs * Es_re) * s + (rp * Ep_re) * p_ref
    E_imag = (rs * Es_im) * s + (rp * Ep_im) * p_ref

    return construct_efield(E_real, E_imag, efield.freq), ref




@wp.func
def sum_efield_at_rx(efield:EField, rx_array:wp.array(dtype=wp.float32, ndim=2), rx_idx: wp.uint8):
    wp.atomic_add(rx_array, rx_idx, 0, efield.re[0])
    wp.atomic_add(rx_array, rx_idx, 1, efield.re[1])
    wp.atomic_add(rx_array, rx_idx, 2, efield.re[2])
    wp.atomic_add(rx_array, rx_idx, 3, efield.im[0])
    wp.atomic_add(rx_array, rx_idx, 4, efield.im[1])
    wp.atomic_add(rx_array, rx_idx, 5, efield.im[2])


@wp.func
def sum_complex_voltage_at_rx(efield: EField, direction: wp.vec3, rx_voltages: wp.array(dtype=wp.float32, ndim=2), rx_idx: wp.uint8,pol_re: wp.vec3,pol_im: wp.vec3):

    direction,e2,e3 = build_transverse_basis(direction, pol_re, pol_im)
    v_re = wp.dot(e2, efield.re) - wp.dot(e3, efield.im)
    v_im = wp.dot(e2, efield.im) + wp.dot(e3, efield.re)


    # v = conj(polarization) * E
    #v_re = wp.dot(pol_re, efield.re) + wp.dot(pol_im, efield.im) 
    #v_im = wp.dot(pol_re, efield.im) - wp.dot(pol_im, efield.re) 
    wp.atomic_add(rx_voltages, rx_idx, 0, v_re) 
    wp.atomic_add(rx_voltages, rx_idx, 1, v_im)

