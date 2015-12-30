import numpy as np
from menpo.feature import ndfeature

from menpodetect import load_dlib_frontal_face_detector

from menpofit.aam import HolisticAAM, LucasKanadeAAMFitter

import menpo.io as mio
import menpo.image as mimg
from pathlib import Path
import matplotlib.pyplot as plt
from menpofit.aam.base import *
from menpo.image import *
from constants import *
from menpo.landmark import *
from menpo.transform import *
from menpofit.aam import LucasKanadeAAMFitter
import pickle
from faceDetection.facedetect import *
from faceDetection.video import *
from faceDetection.common import *

aam = pickle.load(open("aamNew.p", "rb"))
aam_fitter = LucasKanadeAAMFitter(
    aam,
    n_shape=[5, 15],
    n_appearance=[50, 150]
)

# print str(type(image.landmarks))
# t = image.landmarks.group_labels
# print t[0]
# print image.landmarks[0]

while True:
    t = clock()
    img = initCam()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # cv2.imwrite('test.png', img)

    # for i in range(1, 20) :
    detector = load_dlib_frontal_face_detector()
    # image_path = Path(pathToTrainset + 'image_0003.png')

    # image = mio.import_image('test.png')
    image = mimg.Image(gray)
    # image = image.as_greyscale(mode = 'average')
    detector(image)
    result = aam_fitter.fit_from_bb(image,
                                    image.landmarks['dlib_0'].lms,
                                    max_iters=10)
    #       print i

    image.view()
    aam.reference_shape.bounding_box()._view_2d()
    result.final_shape._view_2d()
    dt = clock() - t
    print 'time: %.1f ms' % (dt * 1000)
    # plt.show()
    # plt.close()


'''

det = OpenCVDetector(cascade)
initial = det.__call__(image)


result = aam_fitter.fit_from_shape(image,
                                                                aam.reference_shape.bounding_box())
#                                image.landmarks['PTS'].lms)
#                                max_iters=30)

result.view()
plt.show()
'''
