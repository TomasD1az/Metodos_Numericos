from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

def f2(x1, x2):
    res = (0.7 * np.exp(-(((9 * x1 - 2) ** 2) / 4) - (((9 * x2 - 2) ** 2) / 4))
           + 0.45 * np.exp(-(((9 * x1 + 1) ** 2) / 9) - (((9 * x2 + 1) ** 2) / 5))
           + 0.55 * np.exp(-(((9 * x1 - 6) ** 2) / 4) - (((9 * x2 - 3) ** 2) / 4))
           - 0.01 * np.exp(-(((9 * x1 - 7) ** 2) / 4) - (((9 * x2 - 3) ** 2) / 4)))
    return res

# Definir los rangos de x1 y x2 y crear una malla de puntos (x1, x2)
x1_range = np.linspace(-1, 1, 100)
x2_range = np.linspace(-1, 1, 100)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# Calcular el valor de f2 en cada punto de la malla
z_values = f2(x1_mesh, x2_mesh)

# Crear una figura 3D y Graficar el polinomio de Lagrange como una superficie
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_mesh, x2_mesh, z_values, cmap='viridis')

# Formatear el plot
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f2(x1, x2)')
plt.show()








# #plotting of function f2 and its aproximations
# ax = plt.axes(projection = "3d")

# x2 = np.arange(-1, 1, 0.01)
# y2 = np.arange(-1, 1, 0.01)

# X, Y = np.meshgrid(x2, y2)

# #Linear Interpolation
# linear_inter = interpolate.interp2d(x2, y2, f2(X,Y), kind='linear', bounds_error=False)
# z_linear = linear_inter(x2,y2)

# #Cubic Splines Interpolation (
# cubic_inter = interpolate.interp2d(x2,y2, f2(X,Y), kind = "cubic")
# z_cubic = cubic_inter(x2,y2)

# ground_truth = ax.plot_surface(X,Y,f2(X,Y), cmap = "plasma")
# plot_linear_inter = ax.plot_surface(X,Y,z_cubic)
# #plot_cubic_inter = ax.plot_surface(X,Y,z_cubic)

# ax.plot_wireframe(X,Y, z_cubic)
# ax.set_xlabel("X values")
# ax.set_ylabel("Y values")
# ax.set_zlabel("Z value")
# plt.colorbar(ground_truth)
# plt.show()