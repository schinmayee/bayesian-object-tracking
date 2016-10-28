#!/usr/bin/env python

from PIL import Image
import numpy as np
import math

# polar coordinates to cartesian
# (origin at viewer, where origin for polar cooardinates is)
# with theta from y axis convention
def GetCartesian(r, theta):
    return np.array([r*math.sin(theta),r*math.cos(theta)], np.float32)

# save given boolean array as a black and white image
def SaveImage(array, file_name):
    im = Image.fromarray(array)
    im.save(file_name)

# get error between two images, as an image, and
# number of false negatives and positives
# positive = occlusion, negative = no occlusion
def GetErrorImage(ref, pred):
    res = np.zeros(shape=np.shape(ref), dtype=np.uint8)
    tpr = float(np.sum(~ref & ~pred))/float(np.sum(~ref))
    fpr = float(np.sum(ref & ~pred))/float(np.sum(ref))
    res[~ref & pred] = 125
    res[ref & ~pred] = 255
    return tpr, fpr, res

# get error as mean of diff between actual distance at an angle
# and predicted distance, the error metric takes a max_dist to use
# for false negatives or positives, when one of the data has a reading
# at an angle, but the other one doesn't (since that should be penalized,
# and equivalent to the object being far in the data where it is not there)
def GetAveError(ref, pred, max_dist):
    ref_r = np.copy(ref)
    pred_r = np.copy(pred)
    ref_r[np.where(ref_r == float('inf'))] = max_dist
    pred_r[np.where(pred_r == float('inf'))] = max_dist
    res = np.average(np.absolute(ref_r-pred_r))
    return res
