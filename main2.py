import numpy as np
from pathlib import Path
import menpo.io as mio
import matplotlib.pyplot as plt
from menpo.feature import  ndfeature
from menpo.landmark import ibug_face_68_trimesh, labeller
from menpodetect import load_dlib_frontal_face_detector
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter
#All paths saved in contants file
from constants import *
#Used for writing AAM to a file
import pickle


#from menpowidgets import visualize_fitting_result

try :
	from menpo.feature import fast_dsift
except :
	pass

detector = load_dlib_frontal_face_detector()

def load_image(i):
    i = i.crop_to_landmarks_proportion(0.5)
    if i.n_channels == 3:
        i = i.as_greyscale()
    # This step is actually quite important! If we are using
    # an AAM and a PiecewiseAffine transform then we need
    # to ensure that our triangulation is sensible so that
    # we don't end up with ugly skinny triangles. Luckily,
    # we provide a decent triangulation in the landmarks
    # package.
    labeller(i, 'PTS', ibug_face_68_trimesh)
    return i

training_images_path = Path(pathToTrainset) 
training_images = [load_image(i) for i in mio.import_images(training_images_path, verbose=True)]

aam = HolisticAAM(
    training_images,
    group='ibug_face_68_trimesh',
    scales=(0.5, 1.0),
    diagonal=150,
    max_appearance_components=200,
    max_shape_components=20,
    verbose=True
)

pickle.dump(aam, open("aamNew.p", "wb"))
