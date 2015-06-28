

import skimage
import pylab as pl
from PIL import Image
import numpy as np
from skimage.color import rgb2hsv
from utils_local.utils_local import savetxt_compact


class ColorExtractor(object):
    """

    """

    def __init__(self, img_fname):
        self.img = np.array(Image.open(img_fname))
        print "image_shape", self.img.shape
        self.patch = np.array([], dtype=np.uint8)
        self.patch_size = 0
        self.sample_rgb = np.zeros((1, 3), dtype=np.uint8)
        self.sample_hsv = np.zeros((1, 3), dtype=np.uint8)

        return

    def mk_center_path(self, patch_size=100, patch_show=False):
        """
        returns a patch of patch_size x patch_size in rgb
        """
        if self.img.shape[0] < patch_size or self.img.shape[1] < patch_size:
                patch_size = min(self.img.shape[0:2]) - 4 # 4 is a buffer zone o stay away from the edges of the image


        self.patch_size = patch_size
        # find the center patch
        xcenter = self.img.shape[1] / 2
        ycenter = self.img.shape[0] / 2

        xleft = xcenter - patch_size / 2
        xright = xcenter + patch_size / 2

        ytop = ycenter - patch_size / 2
        ybottom = ycenter + patch_size / 2

        self.patch = self.img[ytop:ybottom, xleft:xright]

        if patch_show:
            print "patch shape", self.patch.shape
            print "xcenter, ycenter", xcenter, ycenter
            print "patch coordinates", xleft, xright, ytop, ybottom

            pl.figure()
            pl.imshow(self.patch)
            pl.show()

        return

    def mk_random_indices(self, n):
        """
        Generate some x and y coordinates at random
        """
        self.x = np.random.choice(self.patch_size, n, replace=False)
        self.y = np.random.choice(self.patch_size, n, replace=True)
        print self.x, self.y

        # print "x", self.x
        # print "y", self.y

    # def _get_random_pixel_vals(self, npixel):
    #     """
    #     From the center 100x100 pixels, get
    #     the value of npixels at random.
    #     Return an array of npixels x 3
    #     Normally the pixel values are in rgb color
    #     """
    #     # choose a pixel
    #     x = np.array([10, 15], dtype=np.intp)
    #     y = np.array([1, 2], dtype=np.intp)
    #     pixel_rgb = self.patch[y, x, :]
    #     print "random pixels rgb", pixel_rgb
    #     print "random pixels shape", pixel_rgb.shape
    #     self.pixel_rgb = pixel_rgb
    #     #r, g, b = pixel_rgb
    #
    #     return

    def get_sample_rgb(self):
        # check that the image has 3 channels. If so, return the patch
        # Otherwise return 0 0 0
        if len(self.patch.shape) == 3:
            self.sample_rgb = self.patch[self.y, self.x, :]
        return self.sample_rgb


    def get_sample_hsv(self):
        # check that the image has 3 channels
        if len(self.patch.shape) == 3:
            patch_hsv = rgb2hsv(self.patch)
            self.sample_hsv = patch_hsv[self.y, self.x, :]
        return self.sample_hsv


    def write_to_file(self, dst_file, color_space='rgb'):
        print "saving file ", dst_file
        if color_space == 'rgb':
            savetxt_compact(dst_file, self.sample_rgb, fmt="%.6g", delimiter=' ')

        if color_space == 'hsv':
            savetxt_compact(dst_file, self.sample_hsv, fmt="%.6g", delimiter=' ')

        return


def read_features_from_file(filename, desc_dim=3):
    """
    Read feature descriptors and return in matrix form.
    desc_dim = 3.  Because color is expressed in rgb or hsv.
    """

    print filename
    f = np.loadtxt(filename)

    if f.shape[0] == 0:
        f = np.zeros((1, desc_dim))
        print "color descriptor not found", filename

    return f


if __name__ == '__main__':

    from data_manager.data_provider import DataProvider

    fname = '../../DATASETS/dress_attributes/data/json/dataset_dress_all_test.json'
    dp = DataProvider(dataset_fname=fname)

    n = 1
    img_paths = dp.get_random_img_paths(n)

    print img_paths

    img_path = '../../DATASETS/dress_attributes/data/images/Wedding/B00ITKOF2W.jpg'
    dst_path = img_path.replace("data/images/",
                                "vis_representation/color_descriptors/rgb/")
    rgb_dst_path = dst_path.replace(".jpg", ".rgb")
    # hsv_dst_path = dst_path.replace(".jpg", ".hsv")


    ce = ColorExtractor(img_path)
    ce.mk_center_path(patch_size=100, patch_show=True)
    ce.mk_random_indices(n=5)

    sample_rgb = ce.get_sample_rgb()
    sample_hsv = ce.get_sample_hsv()

    ce.write_to_file(rgb_dst_path, color_space='rgb')

    print "sample rgb", sample_rgb
    print "sample hsv", sample_hsv

