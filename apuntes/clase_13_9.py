# hacer un runge kuta 4 y resolver el pendulo

import numpy as np
import matplotlib.pyplot as plt

# definir constantes
g = 9.8
t = 0.0
y = 0.0
z = 0.0

def f(u, v):
    return np.array[v, ]

def g(t, y, z, g):
    return -(g)*np.sin(y)

def rk4(t, y, z, h):
    k1 = h*f(t, y, z)
    k2 = h*f(t + h/2, y + k1/2, z + k1/2)
    k3 = h*f(t + h/2, y + k2/2, z + k2/2)
    k4 = h*f(t + h, y + k3, z + k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)