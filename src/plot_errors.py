#!/usr/bin/env python

import argparse, os
import numpy as np
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser("Plot errors")

parser.add_argument('--config', dest='config', type=str, default='config.txt',
                    help='config file containing error directories'
                         'relative to config file')
parser.add_argument('--output', dest='output_file', type=str,
                    help='Output file')

args = parser.parse_args()
if not args.output_file:
    print('Output file name required')
    parser.print_help()
    exit(1)

assert(os.path.isfile(args.config))

# read configuration for plot
config_dir = os.path.dirname(os.path.realpath(args.config))
error_names = list()
error_dirs = list()
with open(args.config) as config:
    for line in config.readlines():
        data = line.split()
        if len(data) != 2:
            continue
        error_names.append(data[0])
        error_dirs.append(data[1])

N = len(error_names)

# read errors
mean = list()
dev = list()
for dir_name in error_dirs:
    file_name = os.path.join(config_dir, dir_name, 'position_error')
    assert(os.path.isfile(file_name))
    data = open(file_name)
    errors = pickle.load(data)
    data.close()
    mean.append(np.mean(errors))
    dev.append(np.std(errors))

#print(mean)
#print(dev)

ind = np.arange(N)  # the x locations for the groups
width = 0.6       # the width of the bars

fig, ax = plt.subplots()
rects = ax.bar(ind, mean, width, color='k', yerr=dev, align='center')

# add some text for labels, title and axes ticks
ax.set_ylabel('Mean error in position estimate')
#ax.set_ylim([0,1.2*max(mean)])
ax.set_title('Error in Position')
ax.set_xticks(ind)
ax.set_xticklabels(error_names)

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.04f' % height,
                ha='center', va='bottom')

autolabel(rects)

plt.savefig(args.output_file)
