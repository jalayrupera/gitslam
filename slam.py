import cv2
import numpy as np
import g2o

from constants import VIDEO_PATH
from frame import Frame, match_frames, denormalize
from pointmap import Map, Point

# Camera Intrinsics

cap = cv2.VideoCapture(VIDEO_PATH)

W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

F = 525

if W > 1024:
    downscale = 1024.0/W
    F *= downscale
    H = int(H * downscale)
    W = 1024
print("using camera %dx%d with F %f" % (W,H,F))

K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])


mapp = Map(W, H)


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0],4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    
    # Linear Triangulation method
    for i, pt in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = pt[0][0] * pose1[2] - pose1[0]
        A[1] = pt[0][1] * pose1[2] - pose1[1]
        A[2] = pt[1][0] * pose2[2] - pose2[0]
        A[3] = pt[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret


def process_frames(img: np.ndarray):
    img: np.ndarray = cv2.resize(img, (W, H))  # noqa: F405

    new_frame = Frame(mapp, img, K, W, H)

    if new_frame.id == 0:
        return
    print("\n*** frame %d ***" % (new_frame.id,))

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    f1.pose = np.dot(Rt, f2.pose)

    for i,idx in enumerate(idx2):
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])

    # triangulate the points we don't have matches for
    good_pts4d = np.array([f1.pts[i] is None for i in idx1])

    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    good_pts4d &= np.abs(pts4d[:, 3]) != 0
    pts4d /= pts4d[:, 3:]  # homogeneous 3-D coords

    print("Adding:  %d points" % np.sum(good_pts4d))

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        
        u,v = int(round(f1.kpus[idx1[i], 0])), int(round(f1.kpus[idx1[i], 1]))
        pt = Point(mapp, p, img[v, u])
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2,v2), color=(255, 0, 0))

    cv2.imshow("Git SLAM", img)

    if new_frame.id >= 4:
        mapp.optimize()

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
