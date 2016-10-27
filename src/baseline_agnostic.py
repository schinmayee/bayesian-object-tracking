#!/usr/bin/env python

import argparse, os
import data

script_dir = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(script_dir, '..', 'data', 'data.t7')

# command line arguments
parser = argparse.ArgumentParser("Shape agnostic baseline, options: ")
parser.add_argument('--data', dest='data_file', metavar='data-file',
                    type=str, default=data_file,
                    help='Input data file to use')
parser.add_argument('--till', dest='till', metavar='last-frame',
                    type=int, default=pow(10,9),
                    help='Run till this frame, and exit on reaching this '+
                         'frame even if these is more data to process')

args = parser.parse_args()
data_file = args.data_file

if not os.path.isfile(data_file):
    print('Please pass a valid input data file')
    parser.print_usage()
    exit(1)

frames = data.TorchData()
frames.ReadFrom(data_file)
