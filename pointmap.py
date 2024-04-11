import OpenGL.GL as gl
import pangolin
import numpy as np
import g2o

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

    # *** Optimizer *** #
    def optimize(self):
        # Init optimizer
        opt = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        opt.set_algorithm(solver)
        robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))

        # Add frames to Graph
        for f in self.frames:
            sbacam = g2o.SBACam(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]))
            sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[2][0], f.K[2][1], 1.0)

            v_se3 = g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id == 0)
            opt.add_vertex(v_se3)

        # add points to frames
        for p in self.points:
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + 0x10000)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)

            for f in p.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                uv = f.kps[f.pts.index(p)]
                edge.set_measurement(uv)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        opt.set_verbose(True)
        opt.initialize_optimization()
        opt.optimize(20)


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
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
