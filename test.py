import menpo.io as mio
import matplotlib.pyplot as plt
from menpofit.aam.base import *
from menpo.image import *
from constants import *
from menpodetect.opencv import *
from menpo.transform import *
from menpofit.aam import LucasKanadeAAMFitter
import pickle

def apply_transform(pointcloud_vector, transform_vector, n_points):
    result_vector = [0 for i in xrange(len(pointcloud_vector))]
    # TODO:  Account for rotation parameters
    for i in xrange(n_points):
        # Odd elements are the x-coordinate
        # Even elements are the y-coordinate
        result_vector[i] = pointcloud_vector[i] * \
            (1 + transform_vector[0]) + transform_vector[2 + (i % 2)]

    return result_vector

def fit_image(image, aam, path_to_cascade):
    det = OpenCVDetector(path_to_cascade)
    print 'bounding box detected'
    initial = det.__call__(image)
    t = aam.reference_shape
    print len(t.points)
    r = AlignmentSimilarity( initial[0], t.bounding_box())
    transform = apply_transform(t.as_vector(), r.as_vector(), 2 * t.n_points)
    t.from_vector_inplace(np.asarray(transform))
#    t.view()
    print "hrer\n"
#    aam.reference_shape = t;
#    aam.reference_shape.view()
    print len(t.points)
    print 'landmark generated'
    fitter = LucasKanadeAAMFitter(aam, n_shape = 0.9, n_appearance = 0.9)
    #print 'fitter created'
    fr = fitter.fit_from_shape(image, t, gt_shape=None)
    #fr = fitter.fit_from_shape(image, i.landmarks['PTS'].lms)
    image.view()
    # initial[0]._view_2d()
    t.bounding_box()._view_2d()
    #t._view_2d()
    fr.final_shape._view_2d()
    plt.show()

if __name__ == '__main__':
    dataPath = pathToLPFW
    facialCascade = cascade
    testImage = 'image_0001.png'
    i = mio.import_image(
        pathToTrainset + testImage).as_greyscale(mode='luminosity')
    aam = pickle.load(open("aam.p", "rb"))
    print 'aam loaded'
    # print aam.reference_shape
    fit_image(i, aam, cascade)