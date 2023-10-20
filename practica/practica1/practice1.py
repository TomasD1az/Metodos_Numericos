from scipy.interpolate import *
import numpy as np
import matplotlib.pyplot as plt

# 1.0 coleccionar todos los datos

data1 = np.loadtxt('data1.csv', delimiter=' ')
x1 = data1[:,0]
y1 = data1[:,1]

# 2.0 Interpolacion
f1 = interp1d(x1,y1,kind='linear')

f2 = CubicSpline(x1,y1)
x_vals= np.linspace(min(x1),max(x1),100)
y_interp = f2(x_vals)

plt.scatter(x1,y1, label='Datos originales')
plt.plot(x1,f1(x1), 'r')
plt.plot(x_vals, y_interp, label='Interpolación cúbica', color='orange')
plt.title('Splines cubicos vs Interpolacion lineal')
plt.legend(['Interpolacion Lineal','Interpolacion Cubica','Datos'])
plt.grid(True)
plt.show()

# 3.0 Interpolacion con splines cubicos