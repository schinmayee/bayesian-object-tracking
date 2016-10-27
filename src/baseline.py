#!/usr/bin/env python

import argparse, os
import data, utils

script_dir = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(script_dir, '..', 'data', 'data.t7')
output_dir = os.path.join(script_dir, '..', 'results')

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
args = parser.parse_args()
data_file = args.data_file
imgs_dir = os.path.join(output_dir, 'images')

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

sequences=None
if args.data_class == 'TorchData':
    sequences=data.TorchData(
        height=args.data_height, width=args.data_width,
        grid_step=args.data_grid_step,
        angle_min=args.data_angle_min, angle_max=args.data_angle_max)

assert(sequences is not None)
sequences.ReadFrom(data_file)
sequences.BuildIndex()

# code to dump input data to files
for i in range(1,20):
    img_path = os.path.join(imgs_dir, 'input_%010d.png' % i)
    in_bool  = sequences.GetStepBoolean(i)
    utils.SaveBooleanImage(in_bool, img_path)
exit(0)

# get bounding cones for all objects
def GetBoundingCones(raw):
    raise NotImplemented('Call not implemented')

# create a list of (object id, bouding cone 1, bounding cone 2) given
# bounding cone data for 2 time steps
# result contains all objects that are there in second step
def MatchBoundingCones(d1, d2):
    raise NotImplemented('Call not implemented')

# compute displacement given bounding cone data for 2 objects
# if object is not seen in previous frame, assume its velocit as zero
def ComputeVelocities(cones):
    raise NotImplemented('Call not implemented')

# predict boolean image for next step given raw data for current step
def PredictNext(raw_cur, d):
    raise NotImplemented('Call not implemented')

# loop over all data, predicting next sequence and evaluating the prediction
# also, save some data for visualization
raw_prv   = None
cones_prv = None
raw_cur   = sequences.GetStepRaw(0)
cones_cur = GetBoundingCones(raw_cur)
for i in range(1,data.num_sequences-1):
    raw_prv   = raw_cur
    cones_prv = cones_cur
    raw_cur   = sequences.GetStepRaw(i)
    cones_cur = GetBoundingCones(raw_cur)
    id_cones  = MatchBoundingCones(cones_prv, cones_nxt)
    v = ComputeVelocities(cones_prev, cones_cur)
    pred = PredictNext(raw_cur, v)
    # evaluate prediction
    # save some results
