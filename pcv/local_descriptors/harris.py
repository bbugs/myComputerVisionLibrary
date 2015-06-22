"""
Harris corner detector.  I did the basic code, but
I skipped p.50-52.
"""

from PIL import Image
import numpy as np
from scipy.ndimage import filters
import pylab as pl
import matplotlib.cm as cm
from scipy.ndimage import filters
from scripts import imtools

def compute_harris_response(im, sigma=3):
    """
    Compute the Harris corner detector response function
    for each pixel in a graylevel image.
    """
    # derivatives
    # x
    imx = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
    # y
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute components of the Harris matrix
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    #print Wdet
    Wtr = Wxx + Wyy
    #print Wtr
    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    """
    Return corners from a Harris response image
    min_dist is the minimum number of pixels separating
    corners and image boundary.
    """
    # find top corner candidates above a threshold
    corner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > corner_threshold) * 1

    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

    # select the best points taking min_distance into account
    filtered_coords = []

    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
                                (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0

    return filtered_coords


def plot_harris_points(image,filtered_coords):
    """
    Plots corners found in image.
    """
    pl.figure()
    pl.gray()
    pl.imshow(image)
    pl.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    pl.axis('off')
    pl.show()


def get_descriptors(image, filtered_coords, wid=5):
    """
    For each point return pixel values around the point
    using a neighbourhood of width 2 * wid + 1. (Assume points are
    extracted with min_distance > wid).
    """
    desc = []
    for coords in filtered_coords:
        patch = image[coords[0] - wid:coords[0] + wid + 1,
        coords[1]-wid:coords[1] + wid + 1].flatten()
        desc.append(patch)
    return desc

def match(desc1, desc2, threshold=0.5):
    """ For each corner point descriptor in the first image,
    select its match to second image using
    normalized cross correlation. """
    n = len(desc1[0])
    # pair-wise distances
    d = -np.ones((len(desc1), len(desc2)))
    for i in range(len(desc1)):
        for j in range(len(desc2)):
        d1 = (desc1[i] - np.mean(desc1[i])) / np.std(desc1[i])
        d2 = (desc2[j] - np.mean(desc2[j])) / np.std(desc2[j])
        ncc_value = sum(d1 * d2) / (n-1)
        if ncc_value > threshold:
            d[i,j] = ncc_value
    ndx = np.argsort(-d)
    matchscores = ndx[:, 0]

    return matchscores

if __name__ == '__main__':

    #fname = '../images_book/empire.jpg'
    #fname = '../data/dress_sample/B004Z2VIBY.jpg'
    path = '../data/pinterest_sample/'
    im_files = imtools.get_imlist(path)
    for fname in im_files:
        im = np.array(Image.open(fname).convert('L'))
        harrisim = compute_harris_response(im)

        # Note: Increasing the threshold makes it harder (less sensitive) to find corners
        filtered_coords = get_harris_points(harrisim, min_dist=6, threshold=0.1)

        plot_harris_points(im, filtered_coords)