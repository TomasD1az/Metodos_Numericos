import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 5, 20)
y1 = np.cos(x)
y2 = np.sin(x)

plt.plot(x, y1, label="cos(x)")
plt.plot(x, y2, label="sin(x)")
plt.legend()
plt.yscale("linear")
plt.show()  # Show the plot