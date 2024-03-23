import cv2
import numpy as np


from constants import NUM_OF_ORBS


class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(NUM_OF_ORBS)

    def extract(self, img:np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=3)
