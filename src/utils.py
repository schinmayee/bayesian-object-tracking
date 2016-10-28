#!/usr/bin/env python

from PIL import Image
import numpy as np

# save given boolean array as a black and white image
def SaveImage(array, file_name):
    im = Image.fromarray(array)
    im.save(file_name)

# get error between two images
def GetError(ref, pred):
    raise NotImplemented('Call not implemented')
