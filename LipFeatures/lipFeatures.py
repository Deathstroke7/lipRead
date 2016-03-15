'''
TODO: 
    Code to return all the lip features
'''
import numpy as np
import menpodetect
from menpodetect.opencv import *
import menpo.image as mimg

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
        print type(img)
        #gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        #image = mimg.Image(gray)
        return img

    def processImage(self, image, aam_fitter):
        detector = load_opencv_frontal_face_detector()
        detector(image)
        result = aam_fitter.fit_from_bb(image,image.landmarks['PTS'].lms, max_iters=10)
        # print type(result)
        # img = result.final_shape
        # img = result.iter_image
        # st = "iter_" + str(result.n_iters)
        # print img.landmarks[st]['n_points']
        # print b.labels        
        # print type(img)
        # print img.landmarks['iter_0']


x = lipFeatures()
y = x.getAAMFitter()

while True:
    z = x.getGrayscaleImage()
    m = x.processImage(z, y)
