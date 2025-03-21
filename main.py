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
