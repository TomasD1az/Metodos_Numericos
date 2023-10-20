import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# Datos originales
tiempo = np.array([2, 5, 8, 12, 15, 20, 25, 30, 32, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
rendimiento = np.array([0.25, 0.10, 0.05, 0.35, 0.02, 0.60, 0.08, 0.70, 0.75, 0.40, 0.90, 0.30, 0.95, 0.15, 1.00, 0.25, 0.80, 0.10, 0.85, 0.05, 0.75, 0.20, 0.90])

# Crear la interpolación con splines cúbicos
cs = CubicSpline(tiempo, rendimiento)

# Puntos para graficar
x_vals = np.linspace(min(tiempo), max(tiempo), 100)
y_interp = cs(x_vals)

# Graficar los datos originales y la interpolación
plt.scatter(tiempo, rendimiento, label='Datos originales')
plt.plot(x_vals, y_interp, label='Interpolación cúbica', color='orange')
plt.xlabel('Tiempo (min)')
plt.ylabel('Rendimiento')
plt.title('Interpolación con Splines Cúbicos')
plt.legend()
plt.grid(True)
plt.show()