"""
SIFT
includes both an interest point detector and a descriptor. The descriptor is very robust
and is largely the reason behind the success and popularity of SIFT. Since its introduction
many alternatives have been proposed with essentially the same type of descriptor.
"""

# the binaries for vlfeat are located in
# '/Users/susanaparis/Documents/packages/vlfeat-0.9.18/bin/maci64/'

from PIL import Image
import os
import numpy as np
import pylab as pl

sift_dir = '/Users/susanaparis/Documents/packages/vlfeat-0.9.18/bin/maci64/'

def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """
    Process an image and save the results in a file.
    """
    if imagename[-3:] != 'pgm':
        # create a pgm file
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
        # the binaries need the image in grayscale and save it as .pgm

    cmmd = str(sift_dir + "./sift " + imagename + " --output=" + resultname + " " + params)
    print cmmd
    os.system(cmmd)
    print 'processed', imagename, 'to', resultname

def read_features_from_file(filename, desc_dim=132):
    """
    Read feature properties and return in matrix form.
    desc_dim = 132.  This is the first 4 dimensions correspond to
    location and scale, the last 128 dimensions correspond to the sift
    descriptor.  A total of 132 dimensions
    """

    print filename
    f = np.loadtxt(filename)

    if f.shape[0] == 0:
        f = np.zeros((1, desc_dim))
        print filename
    return f[:, :4], f[:, 4:]  # feature locations, descriptors

def plot_features(im, locs, circle=False):
    """
    Show image with features. input: im (image as array),
    locs (row, col, scale, orientation of each feature).
    """

    def draw_circle(c, r):
        t = np.arange(0, 1.01, .01) * 2 * np.pi
        x = r * np.cos(t) + c[0]
        y = r * np.sin(t) + c[1]
        pl.plot(x, y, 'b', linewidth=2)

    pl.imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        pl.plot(locs[:, 0], locs[:, 1], 'ob')
    pl.axis('off')

def extract_patch(im, location):
    """
    im is a numpy array
    locs is the location of the sift
    """
    x = location[0]  # x, y center coordinates
    y = location[1]
    r = location[2]  # radius or "scale"
    a = int(x - r)
    b = int(y - r)
    c = int(x + r)
    d = int(y + r)
    box = (a, b, c, d)
    print box
    region = im[a:c, b:d]
    # print dir(region)
    return region


def match(desc1, desc2):
    """
    For each descriptor in the first image,
    select its match in the second image.
    input: desc1 (descriptors for the first image),
    desc2 (same for second image).
    """

    desc1 = np.array([d / np.linalg.norm(d) for d in desc1])
    desc2 = np.array([d / np.linalg.norm(d) for d in desc2])

    dist_ratio = 0.6
    desc1_size = desc1.shape

    matchscores = np.zeros((desc1_size[0]), 'int')
    desc2t = desc2.T  # precompute matrix transpose

    for i in range(desc1_size[0]):
        dotprods = np.dot(desc1[i, :], desc2t)  # vector of dot products
        dotprods = 0.9999 * dotprods

        # inverse cosine and sort, return index for features in second image
        indx = np.argsort(np.arccos(dotprods))

        # check if nearest neighbor has angle less than dist_ratio times 2nd
        if np.arccos(dotprods)[indx[0]] < dist_ratio * np.arccos(dotprods)[indx[1]]:
            matchscores[i] = int(indx[0])

    return matchscores


def appendimages(im1,im2):
    """
    Return a new image that appends the two images side-by-side.
    """

    # select the image with the fewest rows and fill in enough empty rows
    rows1 = im1.shape[0]
    rows2 = im2.shape[0]

    if rows1 < rows2:
        im1 = np.concatenate((im1, np.zeros((rows2-rows1,im1.shape[1]))), axis=0)
    elif rows1 > rows2:
        im2 = np.concatenate((im2, np.zeros((rows1-rows2,im2.shape[1]))), axis=0)
    # if none of these cases they are equal, no filling needed.

    return np.concatenate((im1,im2), axis=1)


def plot_matches(im1, im2, locs1, locs2, matchscores, show_below=True):
    """
    Show a figure with lines joining the accepted matches
    input: im1,im2 (images as arrays), locs1,locs2 (location of features),
    matchscores (as output from 'match'), show_below (if images should be shown below).
    """

    im3 = appendimages(im1,im2)
    if show_below:
        im3 = np.vstack((im3,im3))

    # show image
    pl.imshow(im3)

    # draw lines for matches
    cols1 = im1.shape[1]
    for i, m in enumerate(matchscores):
        if m > 0:
            pl.plot([locs1[i][0], locs2[m][0] + cols1], [locs1[i][1], locs2[m][1]], 'c')
    pl.axis('off')


def match_twosided(desc1,desc2):
    """
    Two-sided symmetric version of match().
    """

    matches_12 = match(desc1, desc2)
    matches_21 = match(desc2, desc1)

    ndx_12 = matches_12.nonzero()[0]

    # remove matches that are not symmetric
    for n in ndx_12:
        if matches_21[int(matches_12[n])] != n:
            matches_12[n] = 0

    return matches_12

if __name__ == '__main__':
    # imname = '../images_book/empire.jpg'
    # im1 = np.array(Image.open(imname).convert('L'))
    # process_image(imname, 'empire.sift')
    # l1, d1 = read_features_from_file('empire.sift')
    # pl.figure()
    # pl.gray()
    # plot_features(im1, l1, circle=True)
    # pl.show()

    # imdir = '../images_book/ukbench/full/'
    # imgs = os.listdir(imdir)
    # for im in imgs[0:1]:
    #     imname = imdir + im
    #     im1 = np.array(Image.open(imname).convert('L'))
    #     process_image(imname, 'ukbench.sift')
    #     l1, d1 = read_features_from_file('ukbench.sift')
    #     pl.figure()
    #     pl.gray()
    #     plot_features(im1, l1, circle=True)
    #     pl.show()


    #

    #dress_dir = '../data/dress_sample/'
    dress_dir = '/Users/susanaparis/Documents/Belgium/IMAGES_plus_TEXT/DATASETS/dress_attributes/data/images/BridesmaidDresses/'
    dresses = [f for f in os.listdir(dress_dir) if not f.startswith('.')]
    for dress in dresses:
        imname = dress_dir + '%s' % dress
        print imname
        #im1 = np.array(Image.open(imname).convert('L'))
        im1 = np.array(Image.open(imname))
        process_image(imname, 'dress.sift')
        l1, d1 = read_features_from_file('dress.sift')
        pl.figure()
        pl.gray()
        plot_features(im1, l1, circle=True)
        pl.show()
