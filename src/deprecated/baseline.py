#!/usr/bin/env python

import argparse, os
import math
import numpy as np
import cPickle as pickle
import data_reader as dt
import utils

script_dir = os.path.dirname(os.path.realpath(__file__))
data_file = os.path.join(script_dir, '..', 'data', 'data.t7')
output_dir = os.path.join(script_dir, '..', 'results')
thresh_angle = 89.75

# command line arguments
parser = argparse.ArgumentParser("Run baseline algorithm, options: ")
parser.add_argument('--till', dest='till', metavar='last-sequence',
                    type=int, default=20,
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
parser.add_argument('--pred_window', dest='pred_window', metavar='pred-window',
                    type=int, default=4,
                    help='Number of steps to predict')
args = parser.parse_args()
data_file   = args.data_file
output_dir  = args.output
imgs_dir    = os.path.join(output_dir, 'images')
plt_dir     = os.path.join(output_dir, 'plot_data')
dump_freq   = args.dump_freq
pred_window = args.pred_window
max_dist = math.sqrt(float(args.data_width*args.data_width)/4 +
                     args.data_height*args.data_height)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert(os.path.isdir(output_dir))
if not os.path.exists(imgs_dir):
    os.mkdir(imgs_dir)
assert(os.path.isdir(imgs_dir))
if not os.path.exists(plt_dir):
    os.mkdir(plt_dir)
assert(os.path.isdir(plt_dir))

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

# save input
def SaveInput(raw, f):
    img_array = data.ConvertToImgArray(raw)
    img_path  = os.path.join(imgs_dir, 'objects_%010d_all.png' %f)
    utils.SaveImage(img_array, img_path)

# save input
def SaveGrouped(raw, objs, f):
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

# save input
def SavePredicted(raw, f, p):
    img_array = data.ConvertToImgArray(raw)
    img_path  = os.path.join(imgs_dir, 'predicted_%010d_%d.png' %(f, p))
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
    obj = [idx[0][0]]
    for i in range(1,len_idx):
        id1, id2 = idx[0][i-1], idx[0][i]
        r1, r2 = raw[id1], raw[id2]
        if id2 > id1 + 1:
            objs.append(obj)
            obj = [id2]
        else:
            r_tang = math.fabs(r1*cos-r2)
            r_perp = r1*sin
            x1,y1 = utils.GetCartesian(r1, data.GetAngle(id1))
            x2,y2 = utils.GetCartesian(r2, data.GetAngle(id2))
            r_diff = math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
            if r_tang/r_perp < thresh or r_diff <= data.grid_step*math.sqrt(2):
                obj.append(id2)
            else:
                objs.append(obj)
                obj = [id2]
    if len(obj) > 0:
        objs.append(obj)
    return objs


# create a list of matched (object id,1 data 1, object id 2, data 2) given
# objects for 2 time steps
# the code will not generate duplicates for objs2, but may generate duplicates
# (matched multiple times) for objs1
def MatchObjects(raw1, raw2, objs1, objs2):
    t_min = math.radians(data.angle_min)
    t_inc = math.radians(data.angle_step)
    # compute centroids and obj size
    ctrs1, ctrs2 = [], []
    for obj in objs1:
        xy = [utils.GetCartesian(raw1[i], data.GetAngle(i)) for i in obj]
        ctrs1.append(np.average(xy,0))
    for obj in objs2:
        xy = [utils.GetCartesian(raw2[i], data.GetAngle(i)) for i in obj]
        ctrs2.append(np.average(xy,0))
    # compute near objects within  2 grid cell distance (x and y)
    thresh = math.sqrt(2*2+2*2)*data.grid_step
    candidates = {}
    for i2 in range(len(ctrs2)):
        candidates[i2] = []
        c2 = ctrs2[i2]
        for i1 in range(len(ctrs1)):
            c1 = ctrs1[i1]
            d = math.sqrt(np.sum(np.power(c1-c2, 2)))
            if d < thresh:
                candidates[i2].append((i1,d))
    # match i2 to i1
    match = {}
    for i2 in range(len(ctrs2)):
        if len(candidates[i2])>0:
            nbr = min(candidates[i2], key=lambda p : p[1])
            nid = nbr[0]
            match[i2]=nid
        else:
            match[i2]=None
    # add all matched/unmatched pairs to result
    result = []
    matched = set()
    for i2 in range(len(ctrs2)):
        i1 = match[i2]
        if i1 is not None:
            result.append((i1,ctrs1[i1],i2,ctrs2[i2]))
            matched.add(i1)
        else:
            result.append((None,None,i2,ctrs2[i2]))
    for i1 in range(len(ctrs1)):
        if i1 not in matched:
            result.append((i1,ctrs1[i1],None,None))
    return result


# compute displacement given matched objects for two steps
# pid deotes previous step, nid denotes next frame
# (pid, nid are 0 or 1 denoting relative position in objs tuples)
def ComputeDisplacement(objs, pid, nid):
    result = {}
    for grp in objs:
        if grp[2*nid] is None:
            continue
        if grp[2*pid] is not None:
            result[grp[2*nid]] = grp[2*nid+1]-grp[2*pid+1]
        else:
            result[grp[2*nid]] = np.array([0,0], dtype=np.float32)
    return result

# predict next step given raw data for current step
def PredictNext(disp, win, objs, raw):
    res = np.ones(shape=[data.num_angles], dtype=np.float32)*float('inf')
    for obj_id in disp.keys():
        d = disp[obj_id]
        obj = objs[obj_id]
        for i in obj:
            xy = utils.GetCartesian(raw[i],data.GetAngle(i)) + d*float(win)
            r = math.sqrt((xy[0]*xy[0]) + xy[1]*xy[1])
            angle = math.degrees(math.atan2(xy[0], xy[1]))
            rid = float(angle-data.angle_min)/data.angle_step
            ridlo = int(math.floor(rid))
            ridhi = int(math.ceil(rid))
            if ridlo >= 0 and ridlo < data.num_angles:
                res[ridlo] = min(res[ridlo], r)
            if ridhi >= 0 and ridhi < data.num_angles:
                res[ridhi] = min(res[ridhi], r)
    return res

# loop over all data, predicting next sequence and evaluating the prediction
# also, save some data for visualization
till = min(data.num_steps, args.till)
raw_prv  = None
objs_prv = None
raw_cur  = data.GetStepRaw(0)
objs_cur = GetObjects(raw_cur)
SaveInput(raw_cur, 0)
#SaveGrouped(raw_cur, objs_cur, 0)
tpr, fpr, itpr, ifpr, mse_out, mse_io = {}, {}, {}, {}, {}, {}
for p in range(1, pred_window+1):
    k = 'step '+str(p)
    tpr[k],fpr[k],itpr[k],ifpr[k],mse_out[k],mse_io[k] = {},{},{},{},{},{}
for i in range(1,till-pred_window):
    if i%dump_freq == 0:
        print('Processing step %d ...' % (i))
    raw_prv  = raw_cur
    objs_prv = objs_cur
    raw_cur  = data.GetStepRaw(i)
    # save some inputs as images
    if i%dump_freq == 0:
        SaveInput(raw_cur, i)
    objs_cur = GetObjects(raw_cur)
    id_objs  = MatchObjects(raw_prv, raw_cur, objs_prv, objs_cur)
    disp = ComputeDisplacement(id_objs, 0, 1)
    img_in  = data.GetStepImgBool(i)
    for p in range(1,pred_window+1):
        k = 'step '+str(p)
        pred = PredictNext(disp, p, objs_cur, raw_cur)
        # save some results
        if i%dump_freq == 0:
            SavePredicted(pred, i, p)
        # evaluate prediction
        img_out = data.GetStepImgBool(i+p)
        img_pred = data.ConvertToImgBool(pred)
        # error
        tpr[k][i], fpr[k][i], err_im = utils.GetErrorImage(ref=img_out, pred=img_pred)
        # sanity check: find difference between consecutive images in data
        itpr[k][i], ifpr[k][i], diff_im = utils.GetErrorImage(ref=img_in, pred=img_out)
        # save error images
        if i%dump_freq == 0:
            utils.SaveImage(err_im,
                            os.path.join(imgs_dir, 'pred_diff_%010d_%d.png' % (i,p)))
            utils.SaveImage(diff_im,
                            os.path.join(imgs_dir, 'cons_diff_%010d_%d.png' % (i,p)))
        # mse
        mse_out[k][i] = utils.GetAveError(raw_cur, pred, max_dist)
        mse_io[k][i]  = utils.GetAveError(raw_cur, data.GetStepRaw(i+1), max_dist)

# save data for plotting later
err_file = open(os.path.join(plt_dir, 'errors'), 'w')
err = { 'true positive rate': { 'baseline' : tpr, 'repeat n' : itpr}, \
        'false positive rate': { 'baseline' : fpr, 'repeat n' : ifpr}, \
        'mean error' : { 'baseline' : mse_out, 'repeat n':mse_io } }
pickle.dump(err, err_file)
err_file.close()
