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
import cv2
import numpy as np

class PerspectiveNPoints(object):
    '''
    PnP = MediaPipePnPLight(width, height)
    PnP.parse_detected_facial_points_2d() # Reset face landmarks
    X = PnP.project_points_used_in_pnp() # For validation
    X = PnP.project_given_points(3d-points) # For any given points
    X = PnP.get_roll_pitch_yaw() # Get facial angles (roll, pitch, yaw)
    '''
    def __init__(self, width: int, height: int,
                 landmark_indices: list = [
                     1, # Nose
                     33, 263, # Left, right eye
                     61, 291, # Left, right mouth
                     199, # Chin
                     168, # between the eyebrows
                     17, # bottom lip
                     101, 330, # Left, right cheek
                     234, 454, # Left, right ear
                 ]):
        # Features of input frame
        self.width = width
        self.height = height

        # Face landmarks (canonical 3D points)
        self.facial_point_file = \
            'canonical_face_model/canonical_face_model.obj'
        self.facial_points_3d = {}
        self._parse_canonical_facial_points_3d()
        self.facial_points_2d = {}

        # Indices for PnP
        self.landmarks_indices = landmark_indices
        self.cam_matrix = np.array([
            (width,     0,  width / 2),
            (    0, width, height / 2),
            (    0,     0,          1),
        ], dtype = np.float32)
        self.distortion_matrix = np.zeros((4, 1))

    def _parse_canonical_facial_points_3d(self):
        logger.debug('Parse canonical facial 3D points.')
        with open(self.facial_point_file, mode = 'r') as f:
            lines = f.readlines()
            for line in lines:
                elements = line.split()
                if elements[0] == 'v':
                    self.facial_points_3d[len(self.facial_points_3d)] = \
                        (float(elements[1]),
                         float(elements[2]),
                         float(elements[3].replace('\n', '')))

    def parse_detected_facial_points_2d(self, landmarks):
        # logger.debug('Parse detected facial 2D points.')
        self.facial_points_2d = {}
        for landmark in landmarks:
            self.facial_points_2d[len(self.facial_points_2d)] = \
                (landmark.x * self.width, landmark.y * self.height)

    def perspective_n_points(self):
        points_3d_cal = np.array([self.facial_points_3d[index] for index in self.landmarks_indices])
        points_2d_cal = np.array([self.facial_points_2d[index] for index in self.landmarks_indices])

        success, rotation, translation = \
            cv2.solvePnP(points_3d_cal, points_2d_cal, self.cam_matrix, self.distortion_matrix,
                         flags = cv2.SOLVEPNP_ITERATIVE) # SOLVEPNP_EPNP or SOLVEPNP_ITERATIVE

        return (success, rotation, translation)

    def project_points_used_in_pnp(self):
        points_3d_cal = np.array([self.facial_points_3d[index] for index in self.landmarks_indices])
        points_2d_cal = np.array([self.facial_points_2d[index] for index in self.landmarks_indices])

        _, rotation, translation = self.perspective_n_points()
        projected_points_2d_cal, _ = \
            cv2.projectPoints(points_3d_cal, rotation, translation, self.cam_matrix, self.distortion_matrix)

        return (points_2d_cal, projected_points_2d_cal)

    def project_given_points(self, points_3d):
        _, rotation, translation = self.perspective_n_points()
        projected_points_2d_cal, _ = \
            cv2.projectPoints(points_3d, rotation, translation, self.cam_matrix, self.distortion_matrix)

        return projected_points_2d_cal
    
    def get_roll_pitch_yaw(self):
        from math import pi, atan2, asin
        _, rotation, _ = self.perspective_n_points()
        R = cv2.Rodrigues(rotation)[0]
        roll = 188 * atan2(-R[2][1], R[2][2]) / pi
        pitch = 180 * asin(R[2][0]) / pi
        yaw = 180 * atan2(-R[1][0], R[0][0]) / pi

        return (roll, pitch, yaw)