import cv2
import numpy as np

from constants import VIDEO_PATH, W, H
from feature_extractor import FeatureExtractor

cap = cv2.VideoCapture(VIDEO_PATH)
fe = FeatureExtractor()


def process_frames(frame: np.ndarray):
    img: np.ndarray = cv2.resize(frame, (W, H))  # noqa: F405

    kp = fe.extract(img)

    for p in kp:
        u, v = map(lambda x: int(round(x)), p[0])
        cv2.circle(img, (u,v), color=(0,255,0), radius=3)

    cv2.imshow("Git SLAM", img)
    cv2.waitKey(1)


def main():
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            process_frames(frame)
        else:
            break

if __name__ == "__main__":
    main()
