
from skimage.color import rgb2gray
from skimage.morphology import disk, closing, opening, square
from skimage.filters import gaussian
from skimage.filters import threshold_otsu
from skimage.filters.rank import enhance_contrast
from skimage import feature
from skimage.morphology import erosion, dilation,area_closing
import numpy as np
from skimage.morphology import watershed, label
from skimage.feature import peak_local_max
from scipy import ndimage

##Helper Methods#######################################
def multi_ero(im, num, element):
    for i in range(num):
        im = erosion(im, element)
    return im
######################################################################

def ObtainForegroundMask(im):
    im = rgb2gray(im)
    gauss = gaussian(im, sigma=4)
    enhanced = enhance_contrast(gauss, disk(10))
    edges = feature.canny(enhanced, sigma=1.1)
    dilated = dilation(edges, disk(7))
    area_closed = area_closing(dilated, 3000)
    multi_eroded = multi_ero(area_closed, 7)
    return multi_eroded


def FindCellLocations(im, mask):
    im = rgb2gray(im)
    gauss = gaussian(im, sigma=4)
    enhanced = enhance_contrast(gauss, disk(10))
    edges = np.logical_and(feature.canny(enhanced, sigma=0.10), mask)
    area_closed = area_closing(dilation(edges, disk(3)), 7000)
    distance_tr = ndimage.morphology.distance_transform_edt(np.logical_xor(area_closed, dilation(edges, disk(3))))
    coordinates = peak_local_max(distance_tr, min_distance=15)
    thresh = threshold_otsu(im)
    binary = im > thresh - 0.04
    c_list = []
    for c in coordinates:
        if binary[c[0], c[1]]:
            continue
        c_list.append(c)
    c = np.array(c_list)
    return c


