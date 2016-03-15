
from pathlib import Path
import menpo.io as mio
from menpo.feature import ndfeature
#from menpodetect import load_dlib_frontal_face_detector
from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter

import os
#All paths saved in contants file
from constants import *
#Used for writing AAM to a file
import pickle

class AAMInstance:

	def trainAAMObject(self):
		try :
			from menpo.feature import fast_dsift
		except :
			pass

		#detector = load_dlib_frontal_face_detector()

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

		pickle.dump(aam, open(AAMFile, "wb"))


	def getAAMObject(self):
		#if os.path.isfile(pathToTrainedAAM + AAMFile) == False :
		#	self.trainAAMObject()

		aam = pickle.load(open(AAMFile, "rb"))
		return aam


	def getAAMFitter(self, aam):
		aam_fitter = LucasKanadeAAMFitter(aam, n_shape=[5, 15], n_appearance=[50, 150])

		return aam_fitter

