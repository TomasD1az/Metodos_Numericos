from scipy.interpolate import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton


# 1.0 coleccionar todos los datos
data1 = np.loadtxt('mnyo_mediciones.csv', delimiter=' ')
data2 = np.loadtxt('mnyo_ground_truth.csv', delimiter=' ')
x1 = data1[:,0]
y1 = data1[:,1]
x2 = data2[:,0]
y2 = data2[:,1]

# 2.0 hacer interpolacion lineal y splines cubicos
time = np.arange(1,len(data1)+1)
spline_x = CubicSpline(time, x1)
spline_y = CubicSpline(time, y1)
x_new = np.linspace(1, len(data1), 100)
y_interpolated = spline_y(x_new)
x_interpolated = spline_x(x_new)

# 2.1 calcular errores y limites
def f1(x):
    return 10;
def f2(x):
    return 3.6 -0.35 * x
f2_y = f2(x1)

def diferencia_limite_igual_10(x):
    return spline_x(x) - 10
def diferencia_curva(x):
    return spline_y(x) -f2(spline_x(x))

# 2.2 calcular intersecciones
x_p1_interseccion = newton(diferencia_curva, x0=6)
y_p1_interesccion = spline_y(x_p1_interseccion)
x_p2_interseccion = newton(diferencia_limite_igual_10, x0=11)
y_p2_interesccion = spline_y(x_p2_interseccion)

print('interseccion 1: ', x_p1_interseccion, y_p1_interesccion)
print('interseccion 2: ', x_p2_interseccion, y_p2_interesccion)
# 3.0 graficar limites e intersecciones
plt.axvline(x=10, color='gray', linestyle='--', label='limites')
plt.plot(x1, f2_y, color='gray', linestyle='--')
plt.plot(x_p1_interseccion, y_p1_interesccion, 'o', color='black')
plt.plot(x_p2_interseccion, y_p2_interesccion, 'o', color='black')
#  3.1 graficar mediciones y ground truth
plt.scatter(x1, y1, c='r')
plt.plot(x2, y2, label='ground truth')
plt.plot(x1, y1, label='interpolacion lineal', color='#FF5733')
plt.plot(x_interpolated, y_interpolated, label='splines cubicos', color='#FFC300')

# 3.2 formatear el plot
plt.title('Interpolación Lineal vs Splines Cúbicos')
plt.ylim(0, 3.5)
plt.legend()
plt.grid(True)
plt.show()