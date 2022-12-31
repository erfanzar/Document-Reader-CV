import cv2 as cv
import numpy as np
import erutils.utils as ut
from erutils.command_line_interface import *
from utils.utils import *

cfg = ut.read_yaml('hyper-parameters.yaml')

if __name__ == "__main__":
    fprint(*(f'{k} : {v}\n' for k, v in cfg.items()))
    frame = cv.resize(cv.imread(cfg['source']), (cfg['width'], cfg['height']))
    frame = to_bw(frame)
    filter_w = cv.Canny(frame, 100, 200)
    print(filter_w.shape)
    ov = filter_w[:, :] > 200
    for v in range(ov.shape[0]):
        for j in range(ov.shape[1]):
            if ov[v, j]:
                print(v, j)
    frame[ov] = [200, 250, 100]
    print(frame)
    while True:
        cv.imshow('window', frame)
        cv.waitKey(1)
        if cv.waitKey(1) == ord('q'):
            break
