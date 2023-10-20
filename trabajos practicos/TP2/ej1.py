import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# definimos las constantes
G = 9.81
L = 2
M = 1
h = 0.5
Tiempo_inicial= 0
Tiempo_final = 10
steps = int((Tiempo_final - Tiempo_inicial)/ h)
theta0 = math.pi / 20
theta0_dot = 0

# definimos funciones
def kinetic_function(theta):
    return 1/2 * M * (L**2) * (theta**2)

def potential_function(theta):
    return -1 * M * G * L * np.cos(theta) + M * G * L

def total_energy_function(theta, theta_dot):
    return kinetic_function(theta_dot) + potential_function(theta)

def pendulum_dinamic(theta):
    return - ((math.sqrt(G / L)) ** 2) * np.sin(theta)

def pendulum_dinamic_linearized(theta):
    return - ((math.sqrt(G / L)) ** 2) * theta

def position(theta):
    return (L *math.sin(theta), - L * math.cos(theta))

def plot_with(title, ylabel):
    plt.suptitle(title)
    plt.title(f"Largo={L}, ϴ= {round(theta0, 4)}, muestras={steps}, h={h}", color='gray', fontsize=10)
    plt.xlabel('Tiempo')
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# metodos numericos
def analitical(t):
    phi = math.acos(theta0 / (theta0 / math.cos(theta0)))
    amplitude = theta0 / math.cos(phi)
    for i in range(steps-1):
        angle = amplitude * math.cos((math.sqrt(G / L)) * t[i] + phi)
        original_pos.append(position(angle))
        original_ang[i] = angle

def euler(f, h, x, y, z):
    for i in range(1, len(y)):
        x[i] = x[i-1] + h * f(y[i-1])
        y[i] = y[i-1] + h * x[i-1]
        z.append(position(y[i]))

def runge_kuta_4(f, angle_velocities, rk4_ang, rk4_pos):
    for i in range(steps-1):
        k1_theta = angle_velocities[i]
        k1_omega = f(rk4_ang[i])

        k2_theta = angle_velocities[i] + 0.5*h*k1_omega
        k2_omega = f(rk4_ang[i] + 0.5*h*k1_theta)

        k3_theta = angle_velocities[i] + 0.5*h*k2_omega
        k3_omega = f(rk4_ang[i] + 0.5*h*k2_theta)

        k4_theta = angle_velocities[i] + h*k3_omega
        k4_omega = f(rk4_ang[i] + h*k3_theta)

        angle_velocities[i+1] = angle_velocities[i] + (1/6)*h*(k1_omega + 2*k2_omega + 2*k3_omega + k4_omega)
        rk4_ang[i+1] = rk4_ang[i] + (1/6)*h*(k1_theta + 2*k2_theta + 2*k3_theta + k4_theta)
        rk4_pos.append(position(rk4_ang[i+1]))

# evaluamos los metodos
def evaluate_energy(angles, method, h):
    time = np.arange(Tiempo_inicial, Tiempo_final, h)
    total_energy = np.zeros(steps)
    kinetic_energy = np.zeros(steps)
    potential_energy = np.zeros(steps)
    for i in range(len(angles)):
        potential_energy[i] = potential_function(angles[i])
        kinetic_energy[i] = kinetic_function(angle_velocities[i])
        total_energy[i] = total_energy_function(angles[i], angle_velocities[i])

    plt.plot(time, total_energy, label='E.T')
    plt.plot(time, potential_energy, label='E.V')
    plt.plot(time, kinetic_energy, label='E.T')
    plot_with(f'Analisis de energías con {method}', 'Energia')
    return total_energy

def evaluate_angle(angles, method, time):
    plt.plot(time, original_ang,'r', label='Solución linealizada')
    plt.plot(time, angles, 'b', label=f'Solucion de {method}')
    plot_with(f'Solucion de ecuacion diferencial con {method}', 'Valor de ϴ''(t)')

# comparamos metodos
def error_analitica(euler_ang, rk4_ang, original_ang, time):
    error_euler = abs(euler_ang - original_ang)
    error_rk4 = abs(rk4_ang - original_ang)
    plt.plot(time, error_euler, label='Error E.E')
    plt.plot(time, error_rk4, label='Error RK4')
    plot_with('Diferencia entre metodos y solucion linealizada', 'Diferencia a solución linealizada')

def compare_angles(euler_ang, rk4_ang, original_ang, time):
    plt.plot(time,original_ang, label='analitica')
    plt.plot(time, euler_ang, label='ee')
    plt.plot(time, rk4_ang, label='rk4')
    plot_with('Comparación entre los distintos métodos', 'Valor del ángulo')

def compare_energies(euler_energy, rk4_energy, time):
    plt.plot(time, rk4_energy, label='RK4')
    plt.plot(time, euler_energy, label='Euler')
    plot_with('Comparación entre los distintos métodos', 'Energia total')

def error_energy(angles, method, step_sizes):
    errors = []
    for step_size in step_sizes:
        euler(pendulum_dinamic, step_size, angle_velocities, euler_ang, euler_pos)
        total_energy = evaluate_energy(euler_ang, 'Euler', step_size)
        error = abs(total_energy[-1] - total_energy[0])
        errors.append(error)

    plt.plot(step_sizes, errors, label=f'Error {method}')
    plot_with('Error de energia', 'Error')


if __name__ == '__main__':
    #variables para analitico
    original_pos = []
    original_ang = np.zeros(steps)
    angle_velocities = np.zeros(steps)
    angle_velocities[0]= theta0_dot

    # variables para analitica (linealizada)
    linearized_ang = np.zeros(steps)
    linearized_pos = []
    linearized_pos.append(position(theta0))
    angle_velocities_linearized = np.zeros(steps)
    angle_velocities_linearized[0] = theta0_dot

    # variables para euler
    euler_ang = np.zeros(steps)
    euler_ang[0] = theta0
    euler_pos = []
    euler_pos.append(position(theta0))
    
    # variables para RK4
    rk4_ang = np.zeros(steps)
    rk4_ang[0] = theta0
    rk4_pos = []
    rk4_pos.append(position(theta0))

    # variables para RK4 (linealizada)
    rk4_ang_linearized = np.zeros(steps)
    rk4_ang_linearized[0] = theta0
    rk4_pos_linearized = []
    rk4_pos_linearized.append(position(theta0))

    time = np.arange(Tiempo_inicial, Tiempo_final, h)
    analitical(time)

    # evaluamos el metodo de euler
    euler(pendulum_dinamic, h, angle_velocities, euler_ang, euler_pos)
    evaluate_angle(euler_ang, 'Euler', time)
    euler_energy = evaluate_energy(euler_ang, 'Euler', h)

    # evaluamos el metodo de runge kuta 4
    runge_kuta_4(pendulum_dinamic, angle_velocities, rk4_ang, rk4_pos)
    evaluate_angle(rk4_ang, 'Runge Kutta 4', time)
    rk4_energy = evaluate_energy(rk4_ang, 'Runge Kutta 4', h)

    # evaluamos el metodo de runge kuta 4 (linealizado)
    runge_kuta_4(pendulum_dinamic_linearized, angle_velocities_linearized, rk4_ang_linearized, rk4_pos_linearized)
    evaluate_angle(rk4_ang_linearized, 'Runge Kutta 4 (linealizado)', time)
    rk4L_energy = evaluate_energy(rk4_ang_linearized, 'Runge Kutta 4 (linealizado)', h)  

    # Graficamos la solución de ambos métodos
    plt.plot(time, rk4_ang, label='RK4')
    plt.plot(time, rk4_ang_linearized, label='RK4 linealizado')
    plot_with('Comparación entre los distintos métodos', 'Valor del ángulo')

    # comparamos los metodos
    error_analitica(euler_ang, rk4_ang, original_ang, time)
    compare_angles(euler_ang, rk4_ang, original_ang, time)
    compare_energies(euler_energy, rk4_energy, time)
    compare_energies(rk4_energy, rk4L_energy, time)

# si te sobra tiempo y tenes ganas podes hacer esto, pero tampoco es obligatorio
# probando distitnots pasos compare la enegria usanndo euler y rk4
# con distintos angulos las energias totales, con rk4 tanto de la linealizada como de la no, todas las no linealizadas van a ser un coseno perfecto, la linealizada se va a agrandar el error