#!/usr/bin/env python

import numpy as np
import cv2

# local modules
from video import create_capture
from common import clock, draw_str

import sys
import getopt


def initCam():
    args, video_src = getopt.getopt(
        sys.argv[1:], '', ['cascade=', 'nested-cascade='])

    try:
        video_src = video_src[0]
    except:
        video_src = 0

    cam = create_capture(
        video_src, fallback='synth:bg=../data/lena.jpg:noise=0.05')

    ret, img = cam.read()
    return img
    
