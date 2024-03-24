import cv2
import numpy as np

from constants import VIDEO_PATH, W, H
from extractor import Extractor

cap = cv2.VideoCapture(VIDEO_PATH)
fe = Extractor()


def process_frames(frame: np.ndarray):
    img: np.ndarray = cv2.resize(frame, (W, H))  # noqa: F405

    matches = fe.extract(img)

    for pt1, pt2 in matches:
        u1, v1 = map(lambda x: int(round(x)), pt1)
        u2, v2 = map(lambda x: int(round(x)), pt2)

        cv2.circle(img, (u1,v1), color=(0,255,0), radius=3)
        cv2.line(img, (u1, v1), (u2,v2), color=(255, 0, 0), thickness=2)

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
