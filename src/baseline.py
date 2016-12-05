#!/usr/bin/env python

import argparse, os
import numpy as np
import pickle
import data_reader

# input and output directories
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, '..', 'data/random')
output_dir = os.path.join(script_dir, '..', 'baseline_output')

parser = argparse.ArgumentParser("Run baseline algorithms")

parser.add_argument('--frames', dest='frames', type=int, required=True,
    help='Number of input frames to process')
parser.add_argument('--log', dest='log_freq', type=int, default=10,
    help='Logging frequency')
parser.add_argument('--data', dest='data_dir', type=str, default=data_dir,
    help='Input data directory')
parser.add_argument('--output', dest='output_dir', type=str, default=output_dir,
    help='Output directory')

# command line arguments
args = parser.parse_args()
if not os.path.exists(args.data_dir):
    print('Input data directory does not exist')
    exit(1)
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
assert(os.path.isdir(args.output_dir))

if __name__ == '__main__':
    pos_error = list()
    pos_rel_error = list()
    # compute errors when object-observation association is known
    for f in range(1,args.frames):
        if f%args.log_freq == 0:
            print('Finished %d frames ...' % f)
        input_file_name_p = os.path.join(args.data_dir, 'state_%08d.txt' %
                                         (f-1))
        data_p = data_reader.ReadStateWithID(input_file_name_p)
        input_file_name = os.path.join(args.data_dir, 'state_%08d.txt' % f)
        data = data_reader.ReadStateWithID(input_file_name)
        pos_error_f = 0
        pos_rel_error_f = 0
        for oid, state in data.items():
            x_true = np.array(state[0], dtype=float)
            x_obs = np.array(state[2], dtype=float)
            pos_error_d = np.sqrt(np.sum(np.power(x_obs - x_true, 2)))
            pos_error_f += pos_error_d
            if oid in data_p:
                x_true_p = np.array(data_p[oid][0], dtype=float)
                disp = np.sqrt(np.sum(np.power(x_true_p - x_true, 2)))
                if disp != 0:
                    pos_rel_error_f += pos_error_d/disp
        pos_error.append(pos_error_f/len(data))
        pos_rel_error.append(pos_rel_error_f/len(data))
    # dump error
    pos_error_fname = os.path.join(args.output_dir, 'position_error')
    with open(pos_error_fname, 'w') as pos_error_file:
        pickle.dump(pos_error, pos_error_file)
    pos_rel_error_fname = os.path.join(args.output_dir, 'pos_rel_error')
    with open(pos_rel_error_fname, 'w') as pos_rel_error_file:
        pickle.dump(pos_rel_error, pos_rel_error_file)
    print(pos_error)
    print(pos_rel_error)
