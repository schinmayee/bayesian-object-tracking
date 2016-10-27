#!/usr/bin/env python
import torchfile

"""
Abstract class representing input data
"""
class Data():
    
    # read data from a file
    def ReadFrom(file_name):
        raise NotImplemented('Call not implemented')

    # get frame
    def GetFrame(i):
        raise NotImplemented('Call not implemented')


"""
Simulated data in torch file format
"""
class TorchData(Data):
    def __init__(self):
        self.file_name = None
        self.data = None

    # read data from a file
    def ReadFrom(self, file_name):
        self.file_name = file_name
        self.data = torchfile.load(file_name)
        print('Total frames read = %i, sensor recordings per frame = %i' %
              (len(self.data), len(self.data[1])))
