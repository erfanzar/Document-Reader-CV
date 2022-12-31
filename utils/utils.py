import cv2 as cv
import numba as nb
import numpy as np


def open_cam(index: int):
    cam = cv.VideoCapture(index)
    return cam


@nb.jit(fastmath=True)
def to_bw(frame):
    for w in range(frame.shape[0]):
        for h in range(frame.shape[1]):
            v = int(np.mean(frame[w, h]))
            frame[w, h] = [v, v, v]
    return frame
