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

class BlurDetectionsNew(object):
    def __init__(self, input_image_name, yolo_architecture = 'yolov8x'):
        self.input_image_name = input_image_name
        self.input_image = cv2.imread(input_image_name)

        self.yolo_architecture = yolo_architecture
        self.yolo_model = YOLO('%s.pt' % (yolo_architecture))

    def variance_of_laplacian(self, image):
        var = cv2.Laplacian(image, cv2.CV_64F).var()
        return var
    
    def decorate_frame(self, frame, results):
        for result in results:
            boxes = result.boxes
            for bbox, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                lefttop = (int(bbox[0]), int(bbox[3]))
                rightbottom = (int(bbox[2]), int(bbox[1]))
                cv2.rectangle(frame, lefttop, rightbottom, (255, 0, 0), 1)
                cv2.putText(frame, '%s (%.4f)' % (cls, conf), lefttop,
                            cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    def main(self):
        # check objects
        results = self.yolo_model(self.input_image, verbose = True)
        print(results)

        self.decorate_frame(self.input_image, results)

        # check blurry
        gray_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        focus_measure = self.variance_of_laplacian(image = gray_image)

        cv2.imshow('Image', self.input_image)
        key = cv2.waitKey(0)

if __name__ == '__main__':
    path = '../Inputs/Images/Kanna_Hashimoto.jpg'
    blur_detector = BlurDetectionsNew(input_image_name = path)
    blur_detector.main()