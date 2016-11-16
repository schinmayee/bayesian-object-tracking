#!/usr/bin/env python

import argparse, os
import numpy as np
import pickle
import random
import predict
import utils

# input and output directories
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, '..', 'data/random')
output_dir = os.path.join(script_dir, '..', 'output')

parser = argparse.ArgumentParser("Generate object, options: ")

parser.add_argument('--frames', dest='frames', type=int, required=True,
    help='Number of input frames to process')
parser.add_argument('--predict', dest='predict_steps', type=int, default=2,
    help='Number of steps to predict at each frame')
parser.add_argument('--max_objects', dest='max_objects', type=int, default=15,
    help='Max number of objects in a frame')
parser.add_argument('--data', dest='data_dir', type=str, default=data_dir,
    help='Input data directory')
parser.add_argument('--output', dest='output_dir', type=str, default=output_dir,
    help='Output directory')
parser.add_argument('--log', dest='log_freq', type=int, default=1,
    help='Logging frequency')

# command line arguments
args = parser.parse_args()
if not os.path.exists(args.data_dir):
    print('Input data directory does not exist')
    exit(1)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
assert(os.path.isdir(args.output_dir))

if __name__ == '__main__':
    # read in parameters for distribution
    parameter_file_name = os.path.join(self.data_dir, 'parameters')
    parameters = dict()
    with open(parameter_file_name) as parameter_file:
        parameters = pickle.load(parameter_file)
    print('Parameters :\n', parameters)
    file_name = os.path.join(self.data_dir, 'state_%08d.txt'%0)
    # read observed position of visible objects in the zero-th frame
    pos_obs = data_reader.ReadObservedStateShuffled(file_name)
    # estimated object velocity in the zero-th frame
    vel_est = predict.InitializeSimpleRandom(parameters, pos_obs)
    # true state of objects in the zero-th frame
    true_state = data_reader.ReadActualState(file_name)
    # compute and save prediction error
    # TODO
    for f in range(1, args.frames):
        # predict state (position and velocity) for predict_steps frames,
        # starting at frame f
        # TODO
        # read observed position of visible objects in frame f
        file_name = os.path.join(self.data_dir, 'state_%08d.txt'%f)
        pos_obs = data_reader.ReadObservedStateShuffled(file_name)
        # associate visible objects in this frame with those from previous frame
        # TODO
        # prune number of objects from this + previous frame to max_objects
        # TODO
        # create obj_id -> state
        # TODO
        # update state using only the newly observed positions
        # TODO
        # read true state (position and velocity) in frame f
        true_state = data_reader.ReadActualState(file_name)
        # compute and save prediction error
        # TODO
