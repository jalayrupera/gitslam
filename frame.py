import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import EssentialMatrixTransform, FundamentalMatrixTransform

from constants import NUM_OF_ORBS

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]


def denormalize(K, pt):
    ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


def pose_rt(R, t):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    return pose


def extract_rt(E):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0,0,1]], dtype=float)
    U, w, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U *= -1.0

    if np.linalg.det(Vt) < 0:
        Vt *= -1.0

    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)

    t = U[:, 2]

    return np.linalg.inv(pose_rt(R, t))


def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret = []
    idx1,idx2 = [], []

    for m, n in matches:
        if m.distance < 0.75*n.distance:
            p1 = f1.kps[m.queryIdx]
            p2 = f2.kps[m.trainIdx]

            # travel less than 10% of diagonal and be within orb distance 32
            if np.linalg.norm((p1-p2)) < 0.1*np.linalg.norm([f1.W, f1.H]) and m.distance < 32:
            # keep around indices
            # TODO: refactor this to not be O(N^2)
                if m.queryIdx not in idx1 and m.trainIdx not in idx2:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)

                    ret.append((p1, p2))

    # no duplicates
    assert(len(set(idx1)) == len(idx1))
    assert(len(set(idx2)) == len(idx2))

    assert len(ret) >= 8
    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # fit matrix
    model, inliers = ransac((ret[:, 0], ret[:, 1]),
                            # EssentialMatrixTransform,
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.02,
                            max_trials=200)
    print("Matches: %d -> %d -> %d -> %d" % (len(f1.des), len(matches), len(inliers), sum(inliers)))

    # ignore outliers
    Rt = extract_rt(model.params)

    # return
    return idx1[inliers], idx2[inliers], Rt


def extract(img:np.ndarray):
    orb = cv2.ORB_create(NUM_OF_ORBS)

    # Detection
    pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)

    # Extraction
    kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    kps, des = orb.compute(img, kps)

    # return pts and desc
    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


class Frame(object):
    def __init__(self, mapp, img, K, W, H):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

        self.kpus, self.des = extract(img)
        self.kps = normalize(self.Kinv, self.kpus)
        self.pts = [None]*len(self.kps)
        self.pose = np.eye(4)

        self.id = len(mapp.frames)
        mapp.frames.append(self)

        self.W = W
        self.H = H
