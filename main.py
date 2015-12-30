import menpo.io as mio
import matplotlib.pyplot as plt
from menpofit.aam.base import *
from menpo.image import *
import pickle

#breaking_bad = mio.import_builtin_asset.breakingbad_jpg()
#breaking_bad = breaking_bad.as_masked()
# breaking_bad.crop_to_landmarks_inplace(boundary=20)
# breaking_bad.constrain_mask_to_landmarks()
# breaking_bad.view(masked=False);
# print(breaking_bad.mask)
# breaking_bad.view_landmarks();
path_to_lfpw = '/home/deathstroke/NSIT/lipRead/'  # change path accordingly
training_images = []
# load landmarked images
for i in mio.import_images(path_to_lfpw + 'trainset/*', verbose=True):
    # crop image
    # print i.__dict__
    # break
    i.crop_to_landmarks_proportion_inplace(0.1)
    # print type(i)
    # convert it to greyscale if needed
    if i.n_channels == 3:
        i = i.as_greyscale(mode='luminosity')
    print type(i._landmarks)
    # append it to the list
    #x = i.__setitem__('PTS', i._landmarks)
    #i = MaskedImage(i.pixels)
    #i.__class__ = MaskedImage
    # print i._view_landmarks_2d()
    training_images.append(i)
#    i.rescale_to_pointcloud();

"""aam = AAM(training_images, verbose=True, diagonal=120)
pickle.dump(aam, open("aam.p", "wb"))
print type(aam)
print aam
# aam.instance().view()
# plt.show()"""
