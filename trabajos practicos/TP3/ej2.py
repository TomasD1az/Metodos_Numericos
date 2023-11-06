import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA


def reducir_dimension_PCA(X, k): # utiliza sklearn
    pca = PCA(n_components=k)
    X_pca = pca.fit_transform(X)
    return X_pca

def matriz_similaridad(X_PCA, d):
    distancias_pca = euclidean_distances(X_PCA, X_PCA)
    matriz_similitud = 1 / (1 + distancias_pca)

    plt.figure(figsize=(8, 6))
    plt.imshow(matriz_similitud, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title('Matriz de Similitud PCA para d=' + str(d), fontsize=14)
    plt.show()

def ordenar_y_plotear_valores_singulares(S, d=None):
    if d is not None:
        singular_values_to_plot = S[:d]
        singular_values_percentage = (singular_values_to_plot / np.sum(S)) * 100
        title = f'Porcentaje de los {d} Valores Singulares más importantes'
    else:
        singular_values_percentage = (S / np.sum(S)) * 100
        title = 'Porcentaje de todos los Valores Singulares'

    bar_plot('Índice de Valor Singular', 'Porcentaje del Valor Singular', 'Suma de los porcentajes: ' + str(np.round(np.sum(singular_values_percentage),3)) + '%', title, range(len(singular_values_percentage)), singular_values_percentage)


def scatter_plot(x_label, y_label, title, PCA_data, original_data=None):
    plt.figure(figsize=(8, 6))
    plt.scatter(original_data[:, 0], original_data[:, 1], label='origniales')
    plt.scatter(PCA_data[:,0], PCA_data[:,1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.show()

def bar_plot (x_label, y_label, title, suptitle, x, height):
    plt.figure(figsize=(8, 6))
    plt.bar(x, height, color='green')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.suptitle(suptitle)
    plt.title(title, fontsize=10, color='gray')
    plt.show()

def main():

    # INTRODUCCION

    data = pd.read_csv('dataset.csv', index_col=0)
    data = data.iloc[1:, 1:]  # Excluye la primera fila y columna
    X = data.values  # Obtener los valores como matriz
    d_values = [2, 4, 6, 20, None]  # Valores de d a probar
    U, S, Vt = np.linalg.svd(X) # Descomposición SVD


    # EJERCICIO 2.1

    for d in d_values:
        X_PCA = reducir_dimension_PCA(X, d)
        scatter_plot('Componente Principal 1', 'Componente Principal 2', 'Proyección de datos reducidos con PCA, con d=' + str(d), X_PCA, X)
        matriz_similaridad(X_PCA, d)

    # Ejericio 2.2
    # Graficar los autovalores de forma ascendente
    for d in d_values:
        ordenar_y_plotear_valores_singulares(S, d)

    # Ejercicio 2.3
    #encontrar B y modelar el problema que minimice la norma

if __name__ == '__main__':
    main()