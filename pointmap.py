import OpenGL.GL as gl
import pangolin
import numpy as np
import g2o

from multiprocessing import Queue, Process
from frame import Frame, pose_rt

LOCAL_WINDOW = 20


class Map(object):
    def __init__(self, W, H):
        self.frames: list[Frame] = []
        self.points: list[Point] = []
        self.state = None
        self.q = Queue()
        self.max_point = 0

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

        local_frames = self.frames[-LOCAL_WINDOW:]

        # Add frames to Graph
        for f in self.frames:
            sbacam = g2o.SBACam(g2o.SE3Quat(f.pose[0:3, 0:3], f.pose[0:3, 3]))
            sbacam.set_cam(f.K[0][0], f.K[1][1], f.K[0][2], f.K[1][2], 1.0)

            v_se3 = g2o.VertexCam()
            v_se3.set_id(f.id)
            v_se3.set_estimate(sbacam)
            v_se3.set_fixed(f.id <= 1 or f not in local_frames)
            opt.add_vertex(v_se3)

        # add points to frames
        PT_ID_OFFSET = 0x1000
        for p in self.points:
            if not any([f in local_frames for f in p.frames]):
                continue
            pt = g2o.VertexSBAPointXYZ()
            pt.set_id(p.id + PT_ID_OFFSET)
            pt.set_estimate(p.pt[0:3])
            pt.set_marginalized(True)
            pt.set_fixed(False)
            opt.add_vertex(pt)

            for f in p.frames:
                edge = g2o.EdgeProjectP2MC()
                edge.set_vertex(0, pt)
                edge.set_vertex(1, opt.vertex(f.id))
                uv = f.kpus[f.pts.index(p)]
                edge.set_measurement(uv)
                edge.set_information(np.eye(2))
                edge.set_robust_kernel(robust_kernel)
                opt.add_edge(edge)

        opt.initialize_optimization()
        opt.optimize(50)
        print(f"Optimizer: {opt.chi2()} units of error")


        # Put frames back
        for f in self.frames:
            est = opt.vertex(f.id).estimate()
            R = est.rotation().matrix()
            t = est.translation()
            f.pose = pose_rt(R, t)

        new_points = []
        # Put points back and cull
        for p in self.points:
            vert = opt.vertex(p.id + PT_ID_OFFSET)
            if vert is None:
                new_points.append(p)
                continue
            est = vert.estimate()

            old_point = len(p.frames) == 2 and p.frames[-1] not in local_frames

            #Reprojection Error
            errs = []
            for f in p.frames:
                uv = f.kpus[f.pts.index(p)]
                proj = np.dot(np.dot(f.K, np.linalg.inv(f.pose)[:3]),
                      np.array([est[0], est[1], est[2], 1.0]))
                proj = proj[0:2] / proj[2]
                errs.append(np.linalg.norm(proj-uv))
            
            #cull
            # if (old_point and np.mean(errs) > 20) or np.mean(errs) > 100:
            #     p.delete()
            #     continue

            p.pt = np.array(est)
            new_points.append(p)      
        self.points = new_points

    def viewer_thread(self):
        self.viewer_init()

        while 1:
            self.viewer_refresh()


    def viewer_init(self):
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
        if not self.q.empty():
            self.state = self.q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        self.dcam.Activate(self.scam)

        if self.state is not None:
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawCameras(self.state[0])

            gl.glPointSize(5)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(self.state[1], self.state[2])

        pangolin.FinishFrame()


    def display(self):
        if self.q is None:
            return

        poses, pts, colors = [], [], []

        for f in self.frames:
            poses.append(f.pose)

        for p in self.points:
            pts.append(p.pt)
            colors.append(p.color)

        self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))



class Point(object):
    def __init__(self, mapp: Map, loc, color):
        self.frames = []
        self.pt = loc
        self.idxs = []
        self.color = np.copy(color)

        self.id = mapp.max_point
        mapp.max_point += 1
        mapp.points.append(self)


    def delete(self):
        for f in self.frames:
            f.pts[f.pts.index(self)] = None
        del self


    def add_observation(self, frame, idx):
        frame.pts[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
