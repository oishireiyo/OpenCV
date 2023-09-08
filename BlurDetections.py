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

# Advanced modules
from ultralytics import YOLO
import cv2
import numpy as np

class BlurDetection(object):
    def __init__(self, input_image_name):
        self.input_image_name = input_image_name
        self.input_image = cv2.imread(input_image_name)

    def variance_of_laplacian(self, image):
        '''
        Compute the Laplacial of the image and then return the focus measure,
        which is simple the variance of the Laplacian.
        '''
        var = cv2.Laplacian(image, cv2.CV_64F).var()
        return var

    def main(self):
        gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        focus_measure = self.variance_of_laplacian(image = gray_image)

        text = 'Not Blurry'
        if focus_measure < 100:
            text = 'Blurry'

        cv2.putText(self.input_image, '{}: {:.2f}'.format(text, focus_measure), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 3)
        cv2.imshow('Image', self.input_image)
        key = cv2.waitKey(0)

if __name__ == '__main__':
    path = '../Inputs/Images/Kanna_Hashimoto.jpg'
    blur_detector = BlurDetection(input_image_name = path)
    blur_detector.main()
