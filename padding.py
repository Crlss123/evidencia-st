import numpy as np

def padding(image, padding):
  filas, col, canales = image.shape
  output = np.zeros((filas+padding*2, col+padding*2, canales), dtype= image.dtype)
  output[padding: padding + filas, padding:padding + col, :] = image

  return output
