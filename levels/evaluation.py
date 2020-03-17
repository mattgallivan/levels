from functools import reduce
import math
import operator

import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity
from scipy.spatial.distance import hamming


def rms(a, b):
    '''
    Compute the root mean square error of two numpy arrays.

    num_tiles = 13
    a = np.random.randint(num_tiles, size=[1, 2])
    b = np.random.randint(num_tiles, size=[1, 2])
    rms(a, b)
    '''
    return np.sqrt(np.mean((a - b) ** 2))


def similarity(a, b):
    '''
    Compute the percentage of elements two numpy arrays have in common.

    num_tiles = 13
    a = np.random.randint(num_tiles, size=[4, 4])
    b = np.random.randint(num_tiles, size=[4, 4])
    similarity(a, b)
    '''
    return 1 - hamming(a.flatten(), b.flatten())


def ssim(original, modified):
    '''
    Compute the structural similarity between an image (ie. 3D numpy array) and its modification.
    Note: Arrays should not be integer data types.

    a = np.random.randint(256, size=[768, 1024, 3]) / float(1)
    b = np.random.randint(256, size=[768, 1024, 3]) / float(1)
    ssim(a, b)
    '''
    return structural_similarity(original, modified, multichannel=True)
