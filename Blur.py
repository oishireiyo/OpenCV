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

# 画像の1点を中心とするカーネルの平均値をその点の値とするぼかし方法
#   image: 画像データ
#   kernel: カーネル
def normal_blur(image: np.ndarray, kernel: tuple[int, int]) -> np.ndarray:
  return cv2.blur(image, kernel)

# カーネルの中心から離れるに連れてガウス関数的に影響が小さくなるようなぼかし方法
#   image: 画像データ
#   kernel: カーネル
#   sigma: ガウス関数の標準偏差に対応。小さくなるにつれ、中心がより重視されるようになる。
def gaussian_blur(image: np.ndarray, kernel: tuple[int, int], sigma: float) -> np.ndarray:
  return cv2.GaussianBlur(image, kernel, sigma)

# 画像の1点を中心とするカーネルの中央値をその点の値とするぼかし方法
#  image: 画像データ
#  kernel_size: カーネルの1辺
def median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
  return cv2.medianBlur(image, kernel_size)

# 画像をぼかしつつ、エッジをうまく残すことができるぼかし方法
#   image: 画像データ
#   kernel_size: カーネルの1辺
#   sigmaColor: ガウス関数の標準偏差に対応。小さくなるにつれ、中心が重視されるようになる。
#   sigmaSpace: 大きいとgaussian_blurに近づき、小さいとノイズ除去効果が小さくなる。
def bilateral_filter(image: np.ndarray, kernel_size: int, sigmaColor: float, sigmaSpace: float) -> np.ndarray:
  return cv2.bilateralFilter(image, kernel_size, sigmaColor, sigmaSpace)

# 画像のノイズを除去しつつ、エッジうまく残すことができるぼかし方法。色がついた画像を扱えるバージョンもある。
#   image: 画像データ
#   hLuminance: 輝度成分のフィルタの平滑化パラメータ。大きいとノイズが減少するが、エッジにも影響する。
#   hColor: 色彩成分のフィルタの平滑化パラメータ。10にしておけば十分。
#   templateWindowSize: カーネルサイズに対応したもの? 7が推奨されている。
#   searchWindowSize: 重み探索の領域サイズ? 21が推奨されている。
def none_local_means_filter(image: np.ndarray, hLuminance: float, templateWindowSize: int=7, searchWindowSize: int=21) -> np.ndarray:
  return cv2.fastNlMeansDenoising(image, None, hLuminance, templateWindowSize, searchWindowSize)

def colored_none_local_means_filter(image: np.ndarray, hLuminance: float, hColor: float=10, templateWindowSize: int=7, searchWindowSize: int=21) -> np.ndarray:
  return cv2.fastNlMeansDenoisingColored(image, None, hLuminance, hColor, templateWindowSize, searchWindowSize)

if __name__ == '__main__':
  from EdgeDetection import convert_to_gray_scale
  image = cv2.imread('assets/aho.jpeg')
  # image = convert_to_gray_scale(image=image)
  # image = none_local_means_filter(image=image, hLuminance=10)
  image = colored_none_local_means_filter(image=image, hLuminance=10)
  cv2.imwrite('outputs/aho.jpeg', image)