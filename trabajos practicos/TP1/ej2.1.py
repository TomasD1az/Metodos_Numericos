from scipy.interpolate import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import math

# 2.1 comparar los tres distintos de metodos con una funcion
# hacemos los casos para puntos equiespaciados y no equiespaciados

# Funciones
def f1(x):
    return (0.05 ** (np.abs(x))) * math.sin(5 * x) + math.tanh(2 * x) + 2

def f2(x1, x2):
    res = (0.7 * np.exp(-(((9 * x1 - 2) ** 2) / 4) - (((9 * x2 - 2) ** 2) / 4))
           + 0.45 * np.exp(-(((9 * x1 + 1) ** 2) / 9) - (((9 * x2 + 1) ** 2) / 5))
           + 0.55 * np.exp(-(((9 * x1 - 6) ** 2) / 4) - (((9 * x2 - 3) ** 2) / 4))
           - 0.01 * np.exp(-(((9 * x1 - 7) ** 2) / 4) - (((9 * x2 - 3) ** 2) / 4)))
    return res

N_POINTS = 10

# Crear puntos de referencia (ground truth)
x1 = np.linspace(-3, 3, N_POINTS)
y1 = [f1(i) for i in x1]

# Crear arreglos para graficar la función de manera precisa
x1_plot = np.linspace(-3, 3, 300)
y1_gt = [f1(i) for i in x1_plot]

# Definir función que interpola y grafica f1 y devuelve los arreglos de interpolación
def interpolate_and_plot(x, y, x_plot, y_plot):
    # Interpolación lineal
    linear_inter = interp1d(x, y, kind='linear', bounds_error=False)
    y_linearinter = linear_inter(x_plot)

    # Interpolación cúbica
    cubic_inter = interp1d(x, y, kind='cubic', bounds_error=False)
    y_cubicinter = cubic_inter(x_plot)

    # Interpolación de Lagrange
    lagrange_inter = lagrange(x, y)
    y_lagrangeinter = lagrange_inter(x_plot)

    # Graficar
    plt.plot(x_plot, y_plot, label='Función Original')
    plt.scatter(x, y, color='red', label='Puntos de Referencia')
    plt.plot(x_plot, y_linearinter, 'orange', label='Interpolación Lineal')
    plt.plot(x_plot, y_cubicinter, 'green', label='Interpolación Cúbica')
    plt.plot(x_plot, y_lagrangeinter, 'black', label='Interpolación de Lagrange')
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return y_linearinter, y_cubicinter, y_lagrangeinter

# Cálculo de errores (f1)

def calc_error(arr, gt):
    error = [abs(gt[i] - arr[i]) for i in range(len(gt))]
    return error

def calc_and_plot_error(x, n_points, x_plot, y_plot, y_linearinter, y_cubicinter, y_lagrangeinter):
    no_error = calc_error(y_plot, y_plot)
    linear_error = calc_error(y_linearinter, y_plot)
    cubic_error = calc_error(y_cubicinter, y_plot)
    lagrange_error = calc_error(y_lagrangeinter, y_plot)

    points = [0] * n_points

    plt.scatter(x, points, color='red', label='Puntos Equiespaciados')
    plt.plot(x_plot, no_error, label='Función Original')
    plt.plot(x_plot, linear_error, 'orange', label='Error Interpolación Lineal')
    plt.plot(x_plot, cubic_error, 'green', label='Error Interpolación Cúbica')
    plt.plot(x_plot, lagrange_error, 'black', label='Error Interpolación de Lagrange')
    plt.xlabel('x')
    plt.ylabel('Error Absoluto')
    plt.legend()
    plt.grid(True)
    plt.show()

y1_linearinter, y1_cubicinter, y1_lagrangeinter = interpolate_and_plot(x1, y1, x1_plot, y1_gt)
calc_and_plot_error(x1, N_POINTS, x1_plot, y1_gt, y1_linearinter, y1_cubicinter, y1_lagrangeinter)

# Método de Chebyshev
def cheb_nodes(n):
    return np.cos(np.pi * (np.arange(1, n + 1) - 0.5) / n)

# Agrupación de nodos utilizando el método de Chebyshev
def fit_nodes(nodes, fin, ini):
    return fin + (ini - fin) * (nodes + 1) / 2

# Definir los puntos
cheb_q = N_POINTS
x1_cheb = fit_nodes(cheb_nodes(cheb_q), 3, -3)
y1_cheb = [f1(i) for i in x1_cheb]

y1_linearinter, y1_cubicinter, y1_lagrangeinter = interpolate_and_plot(x1_cheb, y1_cheb, x1_plot, y1_gt)
calc_and_plot_error(x1_cheb, N_POINTS, x1_plot, y1_gt, y1_linearinter, y1_cubicinter, y1_lagrangeinter)

# Cálculo del error máximo según la cantidad de puntos equiespaciados
def calc_and_plot_max_errors(n):
    e_cubic = []
    e_cheb = []
    x1_domain = np.linspace(-1, 1, 300)
    x2_domain = np.linspace(-1, 1, 300)
    grid_x1, grid_x2 = np.meshgrid(x1_domain, x2_domain)
    for i in range(1, n + 1):  # Comenzar desde 1 para evitar división por cero
        if i < 4:
            # Si hay menos de 4 puntos, no se puede realizar interpolación cúbica
            e_cubic.append(0)  # Agregar 0 como valor de error
            continue
        
        # Generar nodos de interpolación uniformemente distribuidos dentro del dominio
        x1_nodes = np.linspace(-1, 1, i)
        x2_nodes = np.linspace(-1, 1, i)
        x1_nodes, x2_nodes = np.meshgrid(x1_nodes, x2_nodes)
        x1_nodes = x1_nodes.flatten()
        x2_nodes = x2_nodes.flatten()

        # Evaluar la función original en los nodos de interpolación
        z_values = f2(x1_nodes, x2_nodes)

        # Realizar la interpolación en el dominio
        interpolated_values = griddata((x1_nodes, x2_nodes), z_values, (grid_x1, grid_x2), method='cubic')

        # Calcular el error absoluto en cada punto del dominio
        absolute_errors = np.abs(f2(grid_x1, grid_x2) - interpolated_values)

        # Encontrar el valor máximo del error absoluto y guardarlo en la lista
        max_error = np.max(absolute_errors)  # se puede usar np.nanmax para lidiar con "nan" valores
        e_cubic.append(max_error)

        # Puntos no equiespaciados con Chebyshev
        x1_cheb = np.polynomial.chebyshev.chebpts2(i)
        x2_cheb = np.polynomial.chebyshev.chebpts2(i)
        X1_cheb, X2_cheb = np.meshgrid(x1_cheb, x2_cheb)
        Z_cheb = f2(X1_cheb, X2_cheb)

        # Interpolación cúbica con puntos de Chebyshev
        interpolated_values_cheb = griddata((X1_cheb.ravel(), X2_cheb.ravel()), Z_cheb.ravel(), (grid_x1, grid_x2), method='cubic')

        # Calcular el error absoluto para Chebyshev
        absolute_errors_cheb = np.abs(f2(grid_x1, grid_x2) - interpolated_values_cheb)
        max_error_cheb = np.max(absolute_errors_cheb)
        e_cheb.append(max_error_cheb)

    plt.scatter(range(1, n + 1), e_cubic, color='red', label='Puntos Equispaciados')
    plt.xlabel('Cantidad de Puntos')
    plt.ylabel('Error Máximo')
    plt.legend()
    plt.grid(True)
    plt.show()

calc_and_plot_max_errors(N_POINTS)