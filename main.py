import numpy as np
import cv2
from conv import conv
import matplotlib.pyplot as plt
from padding import padding

kernel_nitidez = np.array([
      [-1, -1, -1],
      [-1, 9, -1],
      [-1, -1, -1]
])

kernel_relieve = np.array([
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2]
])

kernel_gauss = (1/16) * np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]])

kernel_bordes = np.array([
  [ 0, -1,  0],
  [-1,  5, -1],
  [ 0, -1,  0]
])

def main():
  path = input("Ingrese la ruta de la imagen: ")
  image = cv2.imread(path, cv2.IMREAD_COLOR_RGB)
  image = conv(image, kernel_nitidez)
  image = conv(image, kernel_gauss)
  image = conv(image, kernel_bordes)
  image = conv(image, kernel_relieve)
  image = padding(image, 1)
  plt.imshow(image)
  plt.axis('off')
  plt.show()

main()
