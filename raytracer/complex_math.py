import warp as wp

@wp.func
def c_add(a_re: wp.float32, a_im: wp.float32, b_re: wp.float32, b_im: wp.float32):
    return a_re + b_re, a_im + b_im

@wp.func
def c_sub(a_re: wp.float32, a_im: wp.float32, b_re: wp.float32, b_im: wp.float32):
    return a_re - b_re, a_im - b_im

@wp.func
def c_mul(a_re: wp.float32, a_im: wp.float32, b_re: wp.float32, b_im: wp.float32):
    return a_re * b_re - a_im * b_im, a_re * b_im + a_im * b_re

@wp.func
def c_div(a_re: wp.float32, a_im: wp.float32, b_re: wp.float32, b_im: wp.float32):
    denom = b_re * b_re + b_im * b_im
    return (a_re * b_re + a_im * b_im) / denom, (a_im * b_re - a_re * b_im) / denom

@wp.func
def c_sqrt(z_re: wp.float32, z_im: wp.float32):
    # principal square root

    mag = wp.sqrt(z_re * z_re + z_im * z_im)
    u = wp.sqrt(wp.float32(0.5) * (mag + z_re))
    v = wp.sqrt(wp.float32(0.5) * (mag - z_re))

    if z_im < 0.0:
        v = -v

    return u, v

@wp.func
def c_vector_rot(z_re:wp.vec3, z_im:wp.vec3, phase:float):

    cph = wp.cos(phase)
    sph = wp.sin(phase)
    re = z_re * cph - z_im * sph
    im = z_re * sph + z_im * cph
    return re, im