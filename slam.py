import cv2
import numpy as np
import g2o

from constants import VIDEO_PATH, W, H
from frame import Frame, match_frames, denormalize

# Camera Intrinsics
F = 290
K = np.array([[F, 0, W//2], [0, F, H//2], [0, 0, 1]])

cap = cv2.VideoCapture(VIDEO_PATH)

frames = []

def process_frames(frame: np.ndarray):
    img: np.ndarray = cv2.resize(frame, (W, H))  # noqa: F405

    new_frame = Frame(img, K)
    frames.append(new_frame)

    if len(frames) <= 1:
        return

    pts, Rt = match_frames(frames[-1], frames[-2])

    for pt1, pt2 in pts:
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2,v2), color=(255, 0, 0))

    cv2.imshow("Git SLAM", img)
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
