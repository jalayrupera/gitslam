import OpenGL.GL as gl
import pangolin
import numpy as np

from multiprocessing import Queue, Process
from frame import Frame

class Map(object):
    def __init__(self, W, H):
        self.frames: list[Frame] = []
        self.points = []
        self.state = None
        self.q = Queue()

        self.W = W
        self.H = H

        p = Process(target=self.viewer_thread, args=())
        p.daemon = True
        p.start()

    def viewer_thread(self):
        self.viewer_init()

        while 1:
            self.viewer_refresh()


    def viewer_init(self):
        print(self.W, self.H)
        pangolin.CreateWindowAndBind('Main', self.W, self.H)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.W, self.H, 420, 420, self.W//2, self.H//2, 0.2, 1000),
            pangolin.ModelViewLookAt(0, -10, -8, 
                                     0, 0, 0, 
                                     0, -1, 0))
        handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -self.W/self.H)
        self.dcam.SetHandler(handler)


    def viewer_refresh(self):
        if self.state is None or not self.q.empty():
            self.state = self.q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(self.state[0])


        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(self.state[1])

        pangolin.FinishFrame()


    def display(self):
        poses, pts = [], []

        for f in self.frames:
            poses.append(f.pose)

        for p in self.points:
            pts.append(p.pt)
        
        self.q.put((np.array(poses), np.array(pts)))



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
