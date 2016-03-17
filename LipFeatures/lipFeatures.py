'''
TODO: 
    Code to return all the lip features
'''
import numpy as np
import menpodetect
from menpodetect.opencv import *
import menpo.image as mimg

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ImageCapture import *
import AAMObject
from constants import *


class lipFeatures:
    def getAAMFitter(self):
        aamObject = AAMObject.AAMInstance()
        aam = aamObject.getAAMObject()
        aam_fitter = aamObject.getAAMFitter(aam)
        return aam_fitter

    def getGrayscaleImage(self):
        img = getImage(0,pathToTestset+'/image_0003.png')
        return img

    def processImage(self, image, aam_fitter):
        detector = load_opencv_frontal_face_detector()
        detector(image)
        result = aam_fitter.fit_from_shape(image,image.landmarks['PTS'].lms, max_iters=10)
        print type(result)
        img = result.final_shape
        image.view()
        img.view()
        plt.show()
        print img
        print img.points


x = lipFeatures()
y = x.getAAMFitter()

z = x.getGrayscaleImage()
m = x.processImage(z, y)
