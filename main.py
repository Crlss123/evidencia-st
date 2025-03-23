import numpy as np
import cv2
import matplotlib.pyplot as plt
from conv import conv
from padding import padding

KERNEL_NITIDEZ = np.array([
    [-1, -1, -1],
    [-1, 9, -1],
    [-1, -1, -1]
])

KERNEL_RELIEVE = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
])

KERNEL_GAUSS = (1 / 16) * np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
])

KERNEL_BORDES = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

def main():
    """Aplica una serie de filtros de convolución a una imagen y la muestra."""
    path = input("Ingrese la ruta de la imagen: ")
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    image = conv(image, KERNEL_NITIDEZ)
    image = conv(image, KERNEL_GAUSS)
    image = conv(image, KERNEL_BORDES)
    image = conv(image, KERNEL_RELIEVE)
    image = padding(image, 1)

    plt.imshow(image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
