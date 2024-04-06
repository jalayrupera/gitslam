import cv2
import numpy as np
import g2o

from constants import VIDEO_PATH, W, H
from frame import Frame, match_frames, denormalize

from multiprocessing import Queue, Process

import OpenGL.GL as gl
import pangolin

# Camera Intrinsics
F = 290
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

cap = cv2.VideoCapture(VIDEO_PATH)


class Map(object):
    def __init__(self):
        self.frames: list[Frame] = []
        self.points = []
        self.state = None
        self.q = Queue()

        p = Process(target=self.viewer_thread, args=(self.q,))
        p.daemon = True
        p.start()


    def viewer_thread(self, q):
        self.viewer_init()

        while 1:
            self.viewer_refresh(q)


    def viewer_init(self):
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(handler)


    def viewer_refresh(self, q):
        if self.state is None or not q.empty():
            self.state = q.get()

        ppts = np.array([d[:3, 3] for d in self.state[0]])
        spts = np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glPointSize(10)
        gl.glColor3f(0.0, 1.0, 0.0)

        poss_arr = [[d[:3, 3] for d in self.state[0]], [d for d in self.state[1]]]

        pangolin.DrawPoints(ppts)

        gl.glPointSize(2)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawPoints(spts)

        pangolin.FinishFrame()


    def display(self):
        poses, pts = [], []

        for f in self.frames:
            poses.append(f.pose)

        for p in self.points:
            pts.append(p.pt)
        
        self.q.put((poses, pts))


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

    # cv2.imshow("Git SLAM", img)

    mapp.display()

    # cv2.waitKey(1)

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
