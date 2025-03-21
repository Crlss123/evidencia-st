import numpy as np

def conv(matriz, kernel):
    filas_matriz, col_matriz, matriz_channels = matriz.shape
    filas_kernel, col_kernel = kernel.shape

    filas_salida = filas_matriz - filas_kernel + 1
    col_salida = col_matriz - col_kernel + 1
    salida = np.zeros((filas_salida, col_salida, matriz_channels))

