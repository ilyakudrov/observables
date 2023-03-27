import math

# find a/r_0 for 5.7 <= beta <= 6.92


def get_r0(beta):
    return math.exp(-1.6804 - 1.7331 * (beta - 6) + 0.7849 * (beta - 6)**2 - 0.4428 * (beta - 6)**3)
