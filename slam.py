import cv2
import numpy as np
import g2o

from constants import VIDEO_PATH, W, H
from frame import Frame, match_frames, denormalize
from map import Map

# Camera Intrinsics
F = 290
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

cap = cv2.VideoCapture(VIDEO_PATH)

mapp = Map()


class Point(object):
    def __init__(self, mapp: Map, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []
        self.id = len(mapp.points)
        mapp.points.append(self)


    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)


def triangulate(pose1, pose2, pts1, pts2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


def process_frames(img: np.ndarray):
    img: np.ndarray = cv2.resize(img, (W, H))  # noqa: F405

    new_frame = Frame(mapp, img, K)

    if new_frame.id == 0:
        return
    
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    # Homogeneous 3d Coords
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]


    good_pts4d = (np.abs(pts4d[:, 3]) > 0.05) & (pts4d[:, 2] > 0)


    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue

        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2,v2), color=(255, 0, 0))

    cv2.imshow("Git SLAM", img)

    mapp.display()

    cv2.waitKey(1)

def main():
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            process_frames(frame)
        else:
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
