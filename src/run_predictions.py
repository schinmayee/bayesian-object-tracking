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
output_dir = os.path.join(script_dir, '..', 'predict_output')
predictor = 'basic'
predictor_types = [
    'basic',
    'unoccluded_nearest',
    'unoccluded_most_likely',
    'occluded_most_likely'
                  ]

parser = argparse.ArgumentParser("Run object tracking using Kalman filtering"
                                " and different data association algorithms")

parser.add_argument('--frames', dest='frames', type=int, required=True,
    help='Number of input frames to process')
parser.add_argument('--predict', dest='predict_steps', type=int, default=2,
    help='Number of steps to predict at each frame')
parser.add_argument('--log', dest='log_freq', type=int, default=1,
    help='Logging frequency')
parser.add_argument('--data', dest='data_dir', type=str, default=data_dir,
    help='Input data directory')
parser.add_argument('--output', dest='output_dir', type=str, default=output_dir,
    help='Output directory')
parser.add_argument('--predictor', dest='predictor', type=str,
                    default=predictor, help='Predictor: ' +
                    ', '.join(predictor_types))

# command line arguments
args = parser.parse_args()
if not os.path.exists(args.data_dir):
    print('Input data directory does not exist')
    exit(1)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
assert(os.path.isdir(args.output_dir))

if __name__ == '__main__':
    kalman = None
    if args.predictor == 'basic':
        kalman = predict.KalmanFilterBasic(args.data_dir, args.output_dir)
    elif args.predictor == 'unoccluded_nearest':
        kalman = predict.KalmanFilterWithAssociation(
            args.data_dir, args.output_dir)
    elif args.predictor == 'unoccluded_most_likely':
        kalman = predict.KalmanFilterWithAssociation(
            args.data_dir, args.output_dir,
            OptimalMatch = predict.SearchOptimalUnoccludedML
        )
    elif args.predictor == 'occluded_most_likely':
        print('Occluded most likley not complete yet')
        kalman = predict.KalmanFilterWithAssociation(
            args.data_dir, args.output_dir, occluded=True,
            OptimalMatch = predict.SearchOptimalOccludedML)
    else:
        print('Predictor should be one of ' + ', '.join(predictor_types))
        exit(1)
    kalman.Run(args.frames, args.log_freq)
