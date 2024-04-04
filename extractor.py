import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform

from constants import NUM_OF_ORBS


def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def extract_rt(E):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0,0,1]], dtype=float)
    U, w, Vt = np.linalg.svd(E)
    
    assert np.linalg.det(U) > 0

    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    
    t = U[:, 2]

    Rt = np.concatenate([R, t.reshape(3,1)], axis=1)

    return Rt

class Extractor(object):
    def __init__(self, K):
        self.orb = cv2.ORB_create(NUM_OF_ORBS)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)


    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]


    def denormalize(self, pt):
        ret = np.dot(self.K, np.array([pt[0], pt[1], 1.0]))
        return int(round(ret[0])), int(round(ret[1]))


    def extract(self, img:np.ndarray):
        # Detection
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        feats = cv2.goodFeaturesToTrack(img, 3000, qualityLevel=0.01, minDistance=3)

        # Extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, desc = self.orb.compute(img, kps)

        # Matching
        ret = []
        if self.last is not None:
            matches = self.matcher.knnMatch(desc, self.last['desc'], k=2)
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))

        Rt = None
        # Filter
        if len(ret) > 0:
            ret = np.array(ret)

            #Cords Normalize
            ret[:, 0, :] = self.normalize(ret[:, 0, :])
            ret[:, 1, :] = self.normalize(ret[:, 1, :])

            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                                EssentialMatrixTransform,
                                                # FundamentalMatrixTransform, #TODO:Replace it with EssentialMatrix
                                                min_samples=8,
                                                # residual_threshold=1,
                                                residual_threshold=0.005,
                                                max_trials=200)

            ret = ret[inliers]
            Rt = extract_rt(model.params)

        self.last = {'kps': kps, 'desc': desc}
        return ret, Rt
