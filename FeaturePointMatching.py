# Standard modules
import os
import sys
import math
import time
import pprint

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
import numpy as np
import cv2

class FeaturePointMatching(object):
    def __init__(self, train_image: str, query_image: str):
        # Features of input frame
        self.train_image = cv2.imread(train_image)
        self.query_image = cv2.imread(query_image)

        # Feature point matching
        self.detector = cv2.AKAZE_create() # AKAZE, BRISK, KAZE, ORB
        self.matcher = cv2.BFMatcher() # BFMatcher, FlannBasedMatcher

    def compute(self):
        '''
        attributes of key_points -->
            pt: キーポイントの座標(ヒント：タプルで(x,y)が戻り値として返ってきます）
            size: キーポイント周辺の重要領域の直径
            angle: 計算されたキーポイントの方向（計算できない場合は -1 ）
            response: 最も強いキーポイントが選択されたときの応答
            octave: キーポイントが抽出されるオクターブ（ピラミッドの階層）
            class_id: オブジェクトクラス
        '''
        train_key_points = self.detector.detect(self.train_image)
        train_key_points, train_descriptions = self.detector.compute(self.train_image, train_key_points)

        query_key_points = self.detector.detect(self.query_image)
        query_key_points, query_descriptions = self.detector.compute(self.query_image, query_key_points)

        '''
        attributes of matches -->
            distance: 特徴量記述子の距離
            queryIdx: クエリ記述子 (match(desc1, desc2) と渡した場合、desc1 のインデックス)
            trainIdx: 学習記述子 (match(desc1, desc2) と渡した場合、desc2 のインデックス)
        '''
        matches = self.matcher.match(train_descriptions, query_descriptions)

        dst = cv2.drawMatches(self.train_image, train_key_points,
                              self.query_image, query_key_points, matches, None)

        cv2.imshow('Final', dst)
        cv2.waitKey(0)

if __name__ == '__main__':
    start_time = time.time()

    train_image = '../Inputs/Images/Kanna_Hashimoto.jpg'
    query_image = '../Inputs/Images/Haruka_Ayase.jpeg'

    matching = FeaturePointMatching(train_image = train_image, query_image = query_image)
    matching.compute()

    end_time = time.time()
    logger.info('Duration: %.4f sec' % (end_time - start_time))
