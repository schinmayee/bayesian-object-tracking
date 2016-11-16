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
