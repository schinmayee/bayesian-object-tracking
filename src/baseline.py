#!/usr/bin/env python

import argparse, os
import math
import numpy as np
import data_reader as dt
import utils

script_dir = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(script_dir, '..', 'data', 'data.t7')
output_dir = os.path.join(script_dir, '..', 'results')
thresh_angle = 89.75

# command line arguments
parser = argparse.ArgumentParser("Shape agnostic baseline, options: ")
parser.add_argument('--till', dest='till', metavar='last-sequence',
                    type=int, default=pow(10,9),
                    help='Run till this sequence, and exit on reaching this '+
                         'sequence even if these is more data to process')
parser.add_argument('--data', dest='data_file', metavar='data-file',
                    type=str, default=data_file,
                    help='Input data file to use')
parser.add_argument('--output', dest='output', metavar='output-dir',
                    type=str, default=output_dir,
                    help='Output directory for saving results')
parser.add_argument('--reader', dest='data_class', metavar='reader-class',
                    type=str, default='TorchData',
                    help='Data class for reading input data')
parser.add_argument('--height', dest='data_height', metavar='pixel-height',
                    type=int, default=51,
                    help='Height of input data in pizels')
parser.add_argument('--width', dest='data_width', metavar='pixel-width',
                    type=int, default=51,
                    help='Width of input data in pizels')
parser.add_argument('--angle_min', dest='data_angle_min', metavar='start-angle',
                    type=int, default=-90,
                    help='Start angle in degrees')
parser.add_argument('--angle_max', dest='data_angle_max', metavar='end-angle',
                    type=int, default=90,
                    help='End angle in degrees')
parser.add_argument('--grid_step', dest='data_grid_step', metavar='grid-step',
                    type=float, default=1,
                    help='Grid cell size')
parser.add_argument('--dump', dest='dump_freq', metavar='dump-freq',
                    type=int, default=10,
                    help='Dump data (images, debug info) every dump-freq steps')
args = parser.parse_args()
data_file = args.data_file
imgs_dir = os.path.join(output_dir, 'images')
dump_freq = args.dump_freq

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert(os.path.isdir(output_dir))
if not os.path.exists(imgs_dir):
    os.mkdir(imgs_dir)
assert(os.path.isdir(imgs_dir))

if not os.path.isfile(data_file):
    print('Please pass a valid input data file')
    parser.print_usage()
    exit(1)

data=None
if args.data_class == 'TorchData':
    data=dt.TorchData(
        height=args.data_height, width=args.data_width,
        grid_step=args.data_grid_step,
        angle_min=args.data_angle_min, angle_max=args.data_angle_max)

assert(data is not None)
data.ReadFrom(data_file)
data.BuildIndex()

# code to save segemented objects
def SaveObjectImages(raw, objs, f):
    img_array = data.ConvertToImgArray(raw)
    img_path  = os.path.join(imgs_dir, 'objects_%010d_all.png' %f)
    utils.SaveImage(img_array, img_path)
    for i in range(len(objs_cur)):
        obj = objs_cur[i]
        img_path  = os.path.join(imgs_dir, 'object_%010d_%03d.png' % (f,i))
        raw_obj = np.ones(shape=np.shape(raw), dtype=np.float32) * \
                  float('inf')
        assert(len(obj) >= 1)
        for i in obj:
            raw_obj[i] = raw[i]
        img_array = data.ConvertToImgArray(raw_obj)
        utils.SaveImage(img_array, img_path)

# segment objects
def GetObjects(raw):
    # each object is a list of indices into raw array
    objs = []  # list of objects to return
    # r1 : prev reading, r2 : next reading
    # if |r1*cos(angle_step)-r2| < r1*sin(angle_step), then same cone
    theta = math.radians(data.angle_step)
    cos = math.cos(theta)
    sin = math.sin(theta)
    thresh = math.tan(math.radians(thresh_angle))
    idx = np.where(raw < float('inf'))
    len_idx = np.shape(idx)[1]
    if len_idx == 0:
        return objs
    # first object
    obj = [idx[0]]
    for i in range(1,len_idx):
        id1, id2 = idx[0][i-1], idx[0][i]
        r1, r2 = raw[id1], raw[id2]
        if id2 > id1 + 1:
            objs.append(obj)
            obj = [id2]
        else:
            r_diff = math.fabs(r1*cos-r2)
            r_perp = r1*sin
            if r_diff/r_perp < thresh:
                obj.append(id2)
            else:
                objs.append(obj)
                obj = [id2]
    objs.append(obj)
    return objs


# create a list of (object id, bouding cone 1, bounding cone 2) given
# bounding cone data for 2 time steps
# result contains all objects that are there in second step
def MatchObjects(d1, d2):
    raise NotImplemented('Call not implemented')

# compute displacement given bounding cone data for 2 objects
# if object is not seen in previous frame, assume its velocit as zero
def ComputeVelocities(objs):
    raise NotImplemented('Call not implemented')

# predict boolean image for next step given raw data for current step
def PredictNext(raw_cur, d):
    raise NotImplemented('Call not implemented')

# loop over all data, predicting next sequence and evaluating the prediction
# also, save some data for visualization
till = min(data.num_steps, args.till)
raw_prv  = None
objs_prv = None
raw_cur  = data.GetStepRaw(0)
objs_cur = GetObjects(raw_cur)
SaveObjectImages(raw_cur, objs_cur, 0)
for i in range(1,till-1):
    if i%dump_freq == 0:
        SaveObjectImages(raw_cur, objs_cur, i)
    raw_prv  = raw_cur
    objs_prv = objs_cur
    raw_cur  = data.GetStepRaw(i)
    objs_cur = GetObjects(raw_cur)
    '''
    id_objs  = MatchObjects(objs_prv, objs_cur)
    v = ComputeVelocities(objs_prev, objs_cur)
    pred = PredictNext(raw_cur, v)
    '''
    # evaluate prediction
    # save some results
