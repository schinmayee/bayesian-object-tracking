#!/usr/bin/env python

import argparse
import os
import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.realpath(__file__))
result_dir = os.path.join(script_dir, '..', 'results')
output_dir = os.path.join(result_dir, '..', 'plots')

parser = argparse.ArgumentParser("Plot average errors, options: ")

parser.add_argument('--data', dest='data_file', metavar='data-file',
                    type=str, required=True,
                    help='Input data file to use for generating plots')
parser.add_argument('--output', dest='out_dir', metavar='out-dir',
                    type=str, required=True,
                    help='Output directory')

args = parser.parse_args()
if output_dir == args.out_dir:
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    assert(os.path.isdir(results_dir))
output_dir = args.out_dir
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert(os.path.isdir(output_dir))

colors=['#3C8D2F', '#AA8839', '#4D2C73']
err_file = open(args.data_file)
errors = pickle.load(err_file)

for typ in errors.keys():
    typ_data = errors[typ]
    # for each error tyype such as false pos, neg and mse
    algs    = sorted(typ_data.keys())
    num_alg = len(algs)
    plt_ave = {}
    plt_std = {}
    x        = None
    # for each algorithm
    for alg in algs:
        alg_data = typ_data[alg]
        if x is None:
            x = alg_data.keys()
        plt_ave[alg] = []
        plt_std[alg] = []
        # for each pred pt (1 in future, 2 in future etc)
        for p in alg_data.keys():  # for each
            # average over all predictions
            p_data = alg_data[p]
            plt_ave[alg].append(np.average([p_data.values()]))
            plt_std[alg].append(np.std([p_data.values()]))
    # now make the plot for this error type
    N = len(x)
    ind = np.arange(N)  # locations on x axis
    width = 0.8/num_alg  # width of bars
    fig, ax = plt.subplots()
    rects = []
    for ai in range(num_alg):  # bar for each algorithm
        alg = algs[ai]
        rects.append(ax.bar(ind+ai*width, plt_ave[alg], width,
                            color=colors[ai%len(colors)], yerr=plt_std[alg]))
    # add some text and labels
    ax.set_xlabel('future step')
    ax.set_ylabel('error')
    ax.set_title(typ)
    ax.set_xticks(ind + width)
    ax.set_xticklabels(x)
    ax.legend(rects, algs)
    fig.savefig(os.path.join(output_dir, typ.replace(' ','-')))
