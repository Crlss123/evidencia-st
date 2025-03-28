import numpy as np

def conv_helper(fragmento, kernel):
    """Aplica la convolución en un fragmento de la imagen con el kernel dado."""
    filas_frag, col_frag, canales_frag = fragmento.shape
    result = np.zeros(canales_frag)

    for canal in range(canales_frag):
        for row in range(filas_frag):
            for col in range(col_frag):
                result[canal] += fragmento[row, col, canal] * kernel[row, col]

    return result

def conv(matriz, kernel):
    """Realiza la convolución de una imagen con un kernel dado."""
    filas_matriz, col_matriz, matriz_channels = matriz.shape
    filas_kernel, col_kernel = kernel.shape

    filas_salida = filas_matriz - filas_kernel + 1
    col_salida = col_matriz - col_kernel + 1
    salida = np.zeros((filas_salida, col_salida, matriz_channels))

    for i in range(filas_salida):
        for j in range(col_salida):
            salida[i, j] = conv_helper(matriz[i:i + filas_kernel, j:j + col_kernel], kernel)

    return np.clip(salida, 0, 255).astype(np.uint8)
