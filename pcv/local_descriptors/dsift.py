from PIL import Image
import os
from numpy import *
from pcv.local_descriptors import sift


def process_image_dsift(imagename, resultname, size=20, steps=10, force_orientation=False, resize=None):
    """
    Process an image with densely sampled SIFT descriptors
    and save the results in a file.
    Optional input: size of features,
                    steps between locations,
                    forcing computation of descriptor orientation
                    (False means all are oriented upwards),
                    tuple for resizing the image.
    """

    im = Image.open(imagename).convert('L')
    if resize != None:
        im = im.resize(resize)
    m, n = im.size
    
    if imagename[-3:] != 'pgm':
        #create a pgm file
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
        # the binaries need the image in grayscale and save it as .pgm

    # create frames and save to temporary file
    scale = size / 3.0
    x, y = meshgrid(range(steps, m, steps), range(steps, n, steps))
    # print x
    # print y
    xx, yy = x.flatten(), y.flatten()
    #print xx


    frame = array([xx, yy, scale * ones(xx.shape[0]), zeros(xx.shape[0])])
    savetxt('tmp.frame', frame.T, fmt='%03.3f')
    
    if force_orientation:
        cmmd = str(sift.sift_dir + "./sift " + imagename + " --output=" + resultname +
                    " --read-frames=tmp.frame --orientations")
    else:
        cmmd = str(sift.sift_dir + "./sift " + imagename + " --output=" + resultname +
                    " --read-frames=tmp.frame")
    os.system(cmmd)
    print 'processed', imagename, 'to', resultname


if __name__ == "__main__":

    #import sift
    from pcv.local_descriptors import sift
    import pylab as pl
    from numpy import *
    from PIL import Image
    # src_file = '../images_book/empire.jpg'
    # dst_file = 'empire.sift'
    # process_image_dsift(src_file, dst_file, size=90, steps=40, force_orientation=True)
    # l, d = sift.read_features_from_file(dst_file)  # feature locations l, and descriptors d
    # # l.shape  (273, 4)
    # # d.shape  (273, 128)
    #
    # im = array(Image.open(src_file))
    # sift.plot_features(im, l, True)
    # pl.show()

    rpath = '/Users/susanaparis/Documents/Belgium/IMAGES_plus_TEXT/DATASETS/dress_attributes/data/images/'
    src_file = rpath + 'BridesmaidDresses/B0057Z09XQ.jpg'
    dst_file = 'B0057Z09XQ.sift'
    process_image_dsift(src_file, dst_file, size=40, steps=20, force_orientation=True)
    l, d = sift.read_features_from_file(dst_file)  # feature locations l, and descriptors d
    # l.shape  (273, 4)
    # d.shape  (273, 128)

    im = array(Image.open(src_file))
    sift.plot_features(im, l, True)
    pl.show()



    # """
    # import nltk.corpus as nc
    # [w for w in nc.words.words() if w.startswith('dilut')]
    # priced
    # praise


    #grump

    #epic
    # """
