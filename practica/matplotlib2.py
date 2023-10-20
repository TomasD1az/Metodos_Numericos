import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Importar para gráficos 3D

# Definir una función de dos variables
def f(x, y):
    return x ** 2 + y ** 2

# Crear valores para x y y
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)  # Crear malla de valores

# Calcular los valores de z utilizando la función
Z = f(X, Y)

# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

# Agregar etiquetas a los ejes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface Plot of f(x, y) = x^2 + y^2')

# Mostrar el gráfico
plt.show()