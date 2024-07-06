import math
import numpy as np

# find a/r_0 for 5.7 <= beta <= 6.92

# hep-lat/0108008
def get_r0(beta):
    return math.exp(-1.6804 - 1.7331 * (beta - 6) + 0.7849 * (beta - 6)**2 - 0.4428 * (beta - 6)**3)

def universal_scaling_function(g2):
    b0 = 11/(4 * math.pi)**2
    b1 = 102/(4 * math.pi)**4
    return (b0*g2)**(-b1/(2*b0**2)) * np.exp(-1./(2*b0) / g2)

def a_hat(g2):
    return universal_scaling_function(g2) * np.reciprocal(universal_scaling_function(g_square(6.)))

def g_square(beta):
    return 6 * np.reciprocal(beta)

# hep-lat/9711003
def get_a_sqrt_sigma(beta):
    return universal_scaling_function(g_square(beta)) * (1 + 0.2731 * a_hat(g_square(beta))**2 - 0.01545 * a_hat(g_square(beta))**4 + 0.01975 * a_hat(g_square(beta))**6) / 0.01364
