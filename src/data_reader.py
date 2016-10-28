#!/usr/bin/env python
import torchfile
import math
import numpy as np

'''
Abstract class representing input data
'''
class Data(object):
    def __init__(self, height, width, grid_step,
                 viewer_x, viewer_y, angle_min, angle_max):
        # initialize members
        assert(width%2 == 1)
        self.height     = height
        self.width      = width
        self.viewer_x   = viewer_x
        self.viewer_y   = viewer_y
        self.grid_step  = grid_step
        self.angle_min  = angle_min
        self.angle_max  = angle_max
        self.num_angles = 0
        self.index      = None
        # file name and actual data
        self.file_name  = None
        self.data       = None
        self.num_steps  = 0
        # distance from viewer/sensor
        self.distance   = np.zeros(shape=[height,width], dtype=np.float32)
        for y in range(height):
            for x in range(width):
                xd = float(x) - self.viewer_x
                yd = float(y)
                self.distance[y,x] = math.sqrt(xd*xd + yd*yd)
        # index into sensor data
        self.angle_step = np.zeros(shape=[height,width], dtype=np.int32)

    # build index into sensor data, from x,y to distance recorded at that angle
    def BuildIndex(self):
        assert(self.data is not None)
        self.num_angles = len(self.data[0])
        self.angle_step = float(self.angle_max-self.angle_min)/ \
                          float(self.num_angles-1)
        self.index = np.zeros(shape=[self.height,self.width], dtype=np.int32)
        for y in range(self.height):
            for x in range(self.width):
                xd = float(x) - self.viewer_x
                yd = float(y)
                angle = math.degrees(math.atan2(xd, yd))
                rdg_id = int(round(float(angle-self.angle_min)/self.angle_step))
                self.index[y][x] = rdg_id
    
    # read data from a file
    def ReadFrom(self, file_name):
        raise NotImplemented('Call not implemented. Override in child class!')

    # get raw data for ith sequence (angle->dist)
    def GetStepRaw(self, i):
        return self.data[i]

    # convert given data to img data ({x,y}->0 if occluded, 255 otherwise)
    def ConvertToImgArray(self, raw):
        rdg = raw[self.index]
        res = np.zeros(shape=[self.height,self.width], dtype=np.uint8)
        res[self.distance + self.grid_step*math.sqrt(0.5) < rdg] = 255
        return res

    # get image data ({x,y}->0 if occluded, 255 otherwise)
    def GetStepImgArray(self, i):
        return self.ConvertToImgArray(self.data[i])


'''
Simulated data in torch file format
'''
class TorchData(Data):
    def __init__(self, height, width, grid_step, angle_min=-90, angle_max=90):
        super(TorchData, self).__init__(
              height=height, width=width, grid_step=grid_step,
              viewer_x = math.ceil(float(width)/2), viewer_y = 0.0,
              angle_min=angle_min, angle_max=angle_max)

    # read data from a file
    # the format of input data is one distance recording per angle,
    # from -180-degrees to 180-degrees
    def ReadFrom(self, file_name):
        self.file_name = file_name
        self.data = torchfile.load(file_name)
        print('Total frames read = %i, sensor recordings per frame = %i' %
              (len(self.data), len(self.data[1])))
        self.num_steps = len(self.data)
