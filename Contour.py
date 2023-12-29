# Standard python modules
import os
import sys
import time

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

# 画像を2値化する
def image_binarization(image: np.ndarray, threshold: int, maxValue: int=255, thresholdType: int=cv2.THRESH_BINARY) -> np.ndarray:
  accepted_thresholdType = [
    cv2.THRESH_BINARY, # threshold 以下の値を0、それ以外の値を maxValue にして2値化を行います。
    cv2.THRESH_BINARY_INV, # threshold 以下の値を maxValue、それ以外の値を0にして2値化を行います。
    cv2.THRESH_TRUNC, # threshold 以下の値はそのままで、それ以外の値を threshold にします。
    cv2.THRESH_TOZERO, # threshold 以下の値を0、それ以外の値はそのままにします。
    cv2.THRESH_TOZERO_INV, # threshold 以下の値はそのままで、それ以外の値を0にします。
    cv2.THRESH_OTSU, # 大津の手法で閾値を自動的に決める場合に指定します。
    cv2.THRESH_TRIANGLE, # ライアングルアルゴリズムで閾値を自動的に決める場合に指定します。
  ]
  if not thresholdType in accepted_thresholdType:
    print('Not accepted thresholdType was given.')

  ret, binary = cv2.threshold(image, threshold, maxValue, thresholdType)

  return ret, binary

# 等高線、等高線の前後関係を取得
def find_contours_and_hierarchy(image: np.ndarray, mode: int=cv2.RETR_EXTERNAL, method: int=cv2.CHAIN_APPROX_NONE) -> np.ndarray:
  accepted_mode = [
    cv2.RETR_EXTERNAL, # 一番外側の白い輪郭のみを取得
    cv2.RETR_LIST, # 白、黒の区別無く全ての輪郭を取得
    cv2.RETR_CCOMP, # 白、黒の輪郭の情報のみを取得
    cv2.RETR_TREE, # 輪郭の階層情報をツリー形式で取得
  ]
  if not mode in accepted_mode:
    print('Not accepted mode was given.')
    sys.exit(1)

  accepted_method = [
    cv2.CHAIN_APPROX_NONE, # 輪郭上の全ての座標を取得する
    cv2.CHAIN_APPROX_SIMPLE, # 縦横斜め45度方向の直線上にある輪郭の点を省略する
    cv2.CHAIN_APPROX_TC89_L1, # 近似できる部分の輪郭を省略
    cv2.CHAIN_APPROX_TC89_KCOS, # 近似できる部分の輪郭を省略
  ]
  if not method in accepted_method:
    print('Not accepted method was given.')
    sys.exit(1)

  contours, hierarchy = cv2.findContours(image, mode, method)
  return contours, hierarchy

if __name__ == '__main__':
  from EdgeDetection import convert_to_gray_scale
  image = cv2.imread('assets/aho.jpeg')
  image = convert_to_gray_scale(image=image)
  _, image = image_binarization(image=image, threshold=100)
  cv2.imwrite('outputs/aho.jpeg', image)
  # contours, _ = find_contours_and_hierarchy(image=image)