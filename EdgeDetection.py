# Standard python modules
import os
import sys
import time
from typing import Union

# Logging
import logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
handler_format = logging.Formatter('%(asctime)s : [%(name)s - %(lineno)d] %(levelname)-8s - %(message)s')
stream_handler.setFormatter(handler_format)
logger.addHandler(stream_handler)

import numpy as np
import cv2

# Return gray scale image
def convert_to_gray_scale(image: np.ndarray) -> np.ndarray:
  return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Return sharpness kernel
def make_sharpness(image: np.ndarray, k: int) -> np.ndarray:
  kernel = np.array([
    [-k / 9,        -k / 9, -k / 9],
    [-k / 9, 1 + 8 * k / 9, -k / 9],
    [-k / 9,        -k / 9, -k / 9],
  ], np.float32)
  return cv2.filter2D(image, -1, kernel).astype('uint8')

# Blurs and image and downsamples it.
def pyramid_down(image: np.ndarray, ndown: int=1) -> np.ndarray:
  temp = image.copy()
  for _ in range(ndown):
    temp = cv2.pyrDown(temp)
  return temp

# Upsamples an image and blurs it.
def pyramid_up(image: np.ndarray, nup: int=1) -> np.ndarray:
  temp = image.copy()
  for _ in range(nup):
    temp = cv2.pyrUp(temp)
  return temp

# Calculates the Laplacian of an image.
def laplacian_edge_detect(image: np.ndarray) -> np.ndarray:
  return cv2.Laplacian(image, cv2.CV_64F)

# Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.
def sobel_edge_detect(image: np.ndarray) -> np.ndarray:
  dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
  dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
  grad = np.sqrt(dx ** 2 + dy ** 2)
  return grad

# Noise Reduction -> Finding Intensity Gradient of the Image -> Non-maximum Sppression -> Hysterrsis Thresholding
def canny_edge_detect(image: np.ndarray, thre1: int, thre2: int) -> np.ndarray:
  return cv2.Canny(image, thre1, thre2, L2gradient=True)

def canny_edge_detect_auto_thres(image: np.ndarray, sigma: float=0.33) -> np.ndarray:
  median_value = np.median(image)
  min_value = int(max(0,   (1.0 - sigma) * median_value))
  max_value = int(max(255, (1.0 + sigma) * median_value))
  print(f'min_value: {min_value}')
  print(f'max_value: {max_value}')
  return cv2.Canny(image, min_value, max_value, L2gradient=True)

if __name__ == '__main__':
  image = cv2.imread('assets/aho.jpeg')
  image = convert_to_gray_scale(image=image)
  image = make_sharpness(image=image, k=3)
  image = canny_edge_detect_auto_thres(image=image)
  cv2.imwrite('outputs/aho.jpeg', image)