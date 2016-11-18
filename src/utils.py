#!/usr/bin/env python

from PIL import Image
import numpy as np
import math

# save given array as an image
def SaveImage(array, file_name):
    im = Image.fromarray(array)
    im.save(file_name)

# read image and return an array
def ReadImage(file_name):
    im = Image.open(file_name)
    return np.array(list(im.getdata()), np.uint8).\
           reshape(im.height, im.width)

