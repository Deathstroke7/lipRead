#!/usr/bin/env python
from video import create_capture
from pathlib import Path
import menpo.io as mio
from menpo.landmark import labeller, face_ibug_68_to_face_ibug_68, face_ibug_68_to_face_ibug_68_trimesh, labeller, face_ibug_68_to_face_ibug_49_trimesh,face_ibug_68_to_face_ibug_66_trimesh, face_ibug_68_to_face_ibug_51_trimesh
import sys
import getopt


def getImage(src,path=None):

    def getImageFromCam():
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


    def getImageFromFile(path):
    
        def load_image(i):
            i = i.crop_to_landmarks_proportion(0.5)
            if i.n_channels == 3:
                i = i.as_greyscale()
            labeller(i, 'PTS', face_ibug_68_to_face_ibug_68)
            return i
        
        image_path = Path(path)
        i =  load_image(mio.import_image(image_path))
        return i



    if src == 0:
        return getImageFromFile(path)
    elif src == 1:
        return getImageFromCam()

