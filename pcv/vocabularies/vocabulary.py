"""
p.189

The class Vocabulary contains a vector of word cluster centers voc
together with the idf values for each word. To train the vocabulary on some set of images, the method
train() takes a list of .sift descriptor files and k, the desired number of words for the

There is also an option of subsampling the training data for the k-means
step which (will take a long time if too many features are used).

"""



import numpy as np
import scipy.cluster.vq as vq

import local_descriptors.sift as sift


class Vocabulary(object):

    def __init__(self, name):
        self.name = name
        self.voc = []  # word cluster centers
        self.idf = []  # idf value for each word
        self.trainingdata = []
        self.nbr_words = 0


    def train(self, featurefiles, k=100, subsampling=10):
        """
        Take a list of .sift descriptor files and k,
        the desired number of words for the vocabulary.

        Train a vocabulary from features in files listed
        in featurefiles (.sift) using k-means with k number of words.
        Subsampling of training data can be used for speedup.
        """

        nbr_images = len(featurefiles)

        # read the features from file (.sift files)
        descr = []  # a list of arrays. Each array is the set of sift descriptors for each iamge
        descr.append(sift.read_features_from_file(featurefiles[0])[1])  # sift descriptors for image 0
        descriptors = descr[0]  # stack all features for k-means
        for i in np.arange(1, nbr_images):
            descr.append(sift.read_features_from_file(featurefiles[i])[1])
            descriptors = np.vstack((descriptors, descr[i]))

        # k-means: last number determines number of runs
        print "computing kmeans"
        self.voc, distortion = vq.kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]
        # voc contains the centroids of each cluster,
        # which are effectively the vocabulary words

        # go through all training images and project on vocabulary
        # print "projecting the images onto the word space"
        # imwords = np.zeros((nbr_images, self.nbr_words))
        # for i in range(nbr_images):
        #     # project assigns to each descriptor the closest vocabulary word
        #     # and returns a histogram () for the whole image i
        #     # descr[i] is an array with all sift descriptors for image i.
        #     imwords[i] = self.project(descr[i])
        #     # imwords is an array nbr_images x nbr_words. It's a term occurrence matrix
        #
        # nbr_occurences = np.sum((imwords > 0) * 1, axis=0)
        #
        # self.idf = np.log( (1.0 * nbr_images) / (1.0 * nbr_occurences + 1))
        # self.trainingdata = featurefiles



    def train_batch(self, descriptors, k=100, subsampling=10):
        """(np array)
        Train on a full bath of dsift or sift descriptors
        """
        print "computing kmeans"
        self.voc, distortion = vq.kmeans(descriptors[::subsampling, :], k, 1)
        self.nbr_words = self.voc.shape[0]





    def project(self, descriptors):
        """(np_array) -> (list, np_array)
        Input: Sift Descriptors for one image
        Output: an array imhist of size nbr_words that contains
        how many times each vocabulary word appears in the image


        Project descriptors on the vocabulary
        to create a histogram of words.

        check where each data point is assigned using
        the vector quantization vq function in the SciPy package.

        """

        # histogram of image words
        imhist = np.zeros(self.nbr_words)
        words, distance = vq.vq(descriptors, self.voc)

        # vq(obs, code_book)
        #         Assigns a code from a code book to each observation. Each
        #         observation vector in the 'M' by 'N' `obs` array is compared with the
        #         centroids in the code book and assigned the code of the closest
        #         centroid.
        #
        #         The features in `obs` should have unit variance, which can be
        #         acheived by passing them through the whiten function.  The code
        #         book can be created with the k-means algorithm or a different
        #         encoding algorithm.



        for w in words:
            imhist[w] += 1  # count how many times each word occurs

        return words, imhist


    def get_words(self, descriptors):
        """
        Convert descriptors to words.
        """

        return vq.vq(descriptors, self.voc)[0]