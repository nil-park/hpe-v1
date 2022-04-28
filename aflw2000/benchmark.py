import os
from scipy.io import loadmat
from os.path import join as ospj
import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

DIR = os.path.dirname(__file__)
DIR_IMAGES = ospj(DIR, 'images')
DIR_LABELS = ospj(DIR, 'labels')


class AFLW2000:

    def __init__(self):
        imfs = sorted([ospj(DIR_IMAGES, f) for f in os.listdir(DIR_IMAGES) if f.endswith('.jpg')])
        lbfs = sorted([ospj(DIR_LABELS, f) for f in os.listdir(DIR_LABELS) if f.endswith('.mat')])
        self.images = np.array([cv2.imread(f) for f in imfs]) # (2000, 450, 450, 3) BGR
        labels = [loadmat(f) for f in lbfs]
        self.poses = np.array([x['Pose_Para'][0] for x in labels]) # (2000, 7) yaw pitch roll cx cy cz scale
        self.pts68 = np.array([x['pt3d_68'] for x in labels]) # (2000, 3, 68) 3d landmark points
        x1 = np.min(self.pts68[:,0], axis=1)
        x2 = np.max(self.pts68[:,0], axis=1)
        y1 = np.min(self.pts68[:,1], axis=1)
        y2 = np.max(self.pts68[:,1], axis=1)
        self.boxes = np.stack([x1, y1, x2, y2], axis=1) # (2000, 4) facial boxes x1 y1 x2 y2
        self.size = len(labels)

    def rotation_mae_in_degree(self, results, skip99=True):
        """ calculate mean average error
            SOTA: https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000

        Parameters:
        results (function): (2000, 3) yaw pitch roll
        skip99 (boolean): skip data degree > 99
        
        Returns:
        ndarray: (4,) MAE: yaw pitch roll mean
        """
        degrees = self.poses[:,:3] * (180 / math.pi)
        if skip99:
            nonskips = np.all(np.abs(degrees) <= 99, axis=1)
            results = results[nonskips]
            degrees = degrees[nonskips]
        pitch, yaw, roll = np.average(np.absolute(degrees - results), axis=0)
        mean = (yaw + pitch + roll) / 3.0
        return np.array([yaw, pitch, roll, mean])

    def rotation_mae_in_degree_with_matrices(self, results, skip99=True):
        """ calculate mean average error
            SOTA: https://paperswithcode.com/sota/head-pose-estimation-on-aflw2000

        Parameters:
        results (function): (2000, 3, 3) rotation matrices
        skip99 (boolean): skip data degree > 99
        
        Returns:
        ndarray: (4,) MAE: yaw pitch roll total
        """
        degrees = self.poses[:,:3] * (180 / math.pi)
        if skip99:
            nonskips = np.all(np.abs(degrees) <= 99, axis=1)
            results = results[nonskips]
            degrees = degrees[nonskips]
        results = np.array(list(map(lambda m: -R.from_matrix(m).as_euler('zyx', degrees=True)[::-1], results)))
        pitch, yaw, roll = np.average(np.absolute(degrees - results), axis=0)
        mean = (yaw + pitch + roll) / 3.0
        return np.array([yaw, pitch, roll, mean])
