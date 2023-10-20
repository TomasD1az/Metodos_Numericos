import numpy as np
import matplotlib.pyplot as plt
import csv

with open ('data.csv', 'r') as cvs:
    lector = csv.reader(cvs)
    valores = []
    for fila in lector:
        fila_valores = [float(valor) for valor in fila]
        valores.append(fila_valores)

array_valores = np.array(valores)
print(array_valores)