import warp as wp

GAUSS_WIDTH = 44.03 * wp.pi / 180
GAUSS_INV_WIDTH2 = 1.0 / (GAUSS_WIDTH * GAUSS_WIDTH)

@wp.func
def calculate_gain(rel_angle:wp.float32):
    # Calculates gain from a fitted gaussian function.
    # The gaussian is fitted to the parameters of the antenna:
    # https://www.getfpv.com/truerc-x-air-2-4ghz-antenna-rhcp.html
    # maxgain = 6.3, mingain = 0.1, halfpowerangle = 37.5deg, isotropicangle = 60deg

    gain = 6.2 * wp.exp(-rel_angle * rel_angle * GAUSS_INV_WIDTH2) + 0.1
    return gain

@wp.kernel
def calculate_gains(directions:wp.array(dtype=wp.vec3), boresight:wp.vec3, amps:wp.array(dtype=wp.float32)):

    tid = wp.tid()
    direction = directions[tid]
    cos_rel_angle = wp.dot(direction, boresight) 
    rel_angle = wp.acos(cos_rel_angle)
    gain = calculate_gain(rel_angle)
    gain_sqrt = wp.sqrt(gain)
    amps[tid] = amps[tid] * gain_sqrt