#!/usr/bin/env python
from video import create_capture

import sys
import getopt


def getImage():
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
