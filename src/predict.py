#!/usr/bin/env python

import pickle
import os
import math
import collections
import numpy as np
import data_reader
import utils

est_color_new = 25
est_color_old = 50

'''
matmul, to make work with numpy <= 1.10.
'''
def matmul(A, B):
    shapeA = np.shape(A)
    shapeB = np.shape(B)
    nshapeA = [0,0]
    nshapeB = [0,0]
    if len(shapeA) == 1:
        nshapeA[0] = 1
        nshapeA[1] = shapeA[0]
    else:
        nshapeA[0]= shapeA[0]
        nshapeA[1]= shapeA[1]
    if len(shapeB) == 1:
        nshapeB[0] = shapeB[0]
        nshapeB[1] = 1
    else:
        nshapeB[0]= shapeB[0]
        nshapeB[1]= shapeB[1]
    R = np.zeros(shape=[nshapeA[0], nshapeB[1]], dtype=float)
    assert(nshapeA[1] == nshapeB[0])
    nA = np.reshape(A, newshape=nshapeA)
    nB = np.reshape(B, newshape=nshapeB)
    for i in range (nshapeA[0]):
        for j in range (nshapeB[1]):
            for k in range(nshapeA[1]):
                R[i,j] += nA[i,k] * nB[k,j]
    if len(shapeA) == 1:
        R = np.reshape(R, newshape=[nshapeB[1],])
    elif len(shapeB) == 1:
        R = np.reshape(R, newshape=[nshapeA[0],])
    return R



'''
Generic predictor class, that reads in a window of data, and predicts upto
some steps in the future.
'''
class Predictor(object):
    '''
    Set parameters, and input/output directories, number of steps to predict,
    and window of data to process.
    '''
    def __init__(self, data_dir, output_dir, predict_steps=1, window=1):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.predict_steps = predict_steps
        self.window = window
        parameter_file_name = os.path.join(data_dir, 'parameters')
        with open(parameter_file_name) as parameter_file:
            self.parameters = pickle.load(parameter_file)
        print('Parameters :\n', self.parameters)
        self.pos_error = list()  # error in estimated position
        self.vel_error = list()  # error in estimated velocity
        self.pos_rel_error = list()

    '''
    Make predictions for next step, and read in data and refine the prediction.
    '''
    def Estimate(self, frame):
        raise NotImplemented('Call')

    '''
    Compute error in prediction.
    '''
    def ComputeError(self):
        frame_pos_error = 0
        frame_vel_error = 0
        frame_pos_rel_error = 0
        for oid, state in self.state.items():
            est = state.GetEstimatedState()
            xe, ve = est[0:2], est[2:4]
            xt, vt = state.GetTrueState()
            x = state.GetTruePositions(2)
            pos_error = np.sqrt(np.sum(np.power(xe-xt, 2)))
            frame_pos_error += pos_error
            displacement = np.sqrt(np.sum(np.power(x[0]-x[1], 2)))
            if displacement != 0:
                frame_pos_rel_error += pos_error/displacement
            frame_vel_error += np.sqrt(np.sum(np.power(ve-vt, 2)))
        num_objects = len(self.state)
        self.pos_error.append(frame_pos_error/num_objects)
        self.vel_error.append(frame_vel_error/num_objects)
        self.pos_rel_error.append(frame_pos_rel_error/num_objects)

    '''
    Save error image.
    '''
    def SaveEstimateAsImage(self, frame):
        file_name = os.path.join(self.data_dir, 'state_%08d.png' % frame)
        im_arr = utils.ReadImage(file_name)
        for oid, state in self.state.items():
            state.SetImage(im_arr)
        out_file_name = os.path.join(self.output_dir, \
                                    'estimate_%08d.png' % frame)
        utils.SaveImage(im_arr, out_file_name)

    '''
    Run predictor.
    '''
    def Run(self, frames, log_freq):
        for f in range(frames):
            if f%log_freq == 0:
                print('Finished %d frames ...' %f)
            self.Estimate(f)
            self.ComputeError()
            if f%log_freq == 0:
                self.SaveEstimateAsImage(f)
        pos_error_fname = os.path.join(self.output_dir, 'position_error')
        with open(pos_error_fname, 'w') as pos_error_file:
            pickle.dump(self.pos_error, pos_error_file)
        vel_error_fname = os.path.join(self.output_dir, 'velocity_error')
        with open(vel_error_fname, 'w') as vel_error_file:
            pickle.dump(self.vel_error, vel_error_file)
        pos_rel_error_fname = os.path.join(self.output_dir, 'pos_rel_error')
        with open(pos_rel_error_fname, 'w') as pos_rel_error_file:
            pickle.dump(self.pos_rel_error, pos_rel_error_file)
        print(self.pos_error)
        print(self.vel_error)
        print(self.pos_rel_error)

'''
Object state, to use for Kalman filtering.
'''
class KalmanObjectState(object):
    zv2 = np.zeros(shape=[2,], dtype=float)
    zv4 = np.zeros(shape=[4,], dtype=float)
    eye4 = np.eye(4, dtype=float)

    def  __init__(self, x_obs=zv2, x_est=zv4,
                  x_pred=zv4, x_cov=eye4,
                  x_true=None, v_true=None, done=False,
                  state_window=5):
        self.x_obs = x_obs    # observed position
        self.x_est = x_est    # estimated [pos:vel]
        self.x_pred = x_pred  # predicted [pos:vel]
        self.x_cov = x_cov    # state covariance (predicted/estimated)
        self.done = done
        self.new_track = True
        self.x_true = collections.deque(maxlen=state_window)
        self.v_true = collections.deque(maxlen=state_window)
        self.unmatched_counter = 0

    def SetTrueState(self, x_true, v_true):
        self.x_true.append(x_true)
        self.v_true.append(v_true)

    def GetTrueState(self):
        return (self.x_true[-1], self.v_true[-1])

    def GetTruePositions(self, num_steps):
        xlen = len(self.x_true)
        return [self.x_true[i] for i in range(xlen-num_steps, xlen)]

    def SetObservation(self, x_obs):
        self.x_obs = x_obs

    def GetObservation(self):
        return self.x_obs

    def SetPredictedState(self, x_pred):
        self.x_pred = x_pred
        self.x_est  = x_pred  # record estimate as predicted in case no update

    def GetPredictedState(self):
        return self.x_pred

    def SetStateCov(self, x_cov):
        self.x_cov = x_cov

    def GetStateCov(self):
        return self.x_cov

    def SetEstimatedState(self, x_est):
        self.x_est = x_est

    def GetEstimatedState(self):
        return self.x_est

    def MarkDone(self):
        self.done = True

    def ResetDone(self):
        self.done = False

    def IsDone(self):
        return self.done

    def SetNewTrack(self):
        self.new_track = True

    def SetOldTrack(self):
        self.new_track = False

    def RecordMatched(self):
        self.unmatched_counter = 0

    def RecordUnmatched(self):
        self.unmatched_counter += 1

    def NumTimesUnmatched(self):
        return self.unmatched_counter

    def SetImage(self, im_arr):
        num = np.shape(im_arr)
        delta = [1.0/n for n in num]
        xc = np.round(self.x_est[0:2]/delta)
        w = math.ceil(num[0]/100.0)
        width = 1
        if self.new_track:
            width = 1.5
        min_pt = np.array(xc - width*w, dtype=int)
        max_pt = np.array(xc + width*w, dtype=int)
        x_min = max(int(min_pt[0]), 0)
        y_min = max(int(min_pt[1]), 0)
        x_max = min(int(max_pt[0]), num[0])
        y_max = min(int(max_pt[1]), num[1])
        est_color = est_color_old
        if self.new_track:
            est_color = est_color_new
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                im_arr[i,j] = int(est_color)

    def Predict(self, F, Q):
        xp = matmul(F, self.GetEstimatedState())
        xcov = matmul(matmul(F, self.GetStateCov()), np.transpose(F)) + Q
        return xp, xcov

    def Update(self, obs, H, R):
        xp = self.GetPredictedState()
        xcov = self.GetStateCov()
        y = obs - matmul(H, xp)
        S = matmul(matmul(H, xcov), np.transpose(H)) + R
        Sinv = np.linalg.inv(S)
        K = matmul(matmul(xcov, np.transpose(H)), Sinv)
        xe = xp + matmul(K, y)
        KH = matmul(K, H)
        xcov = matmul(np.eye(np.shape(KH)[0]) - KH, self.GetStateCov())
        return xe, xcov


'''
Generic Kalman filter, predicts state for next step, and then refines the
estimate using observations from the next step.
Object association and state intialization for new objects is left to child
classes.
'''
class KalmanFilterGeneric(Predictor):
    '''
    The init method takes in input data directory and output directory.
    '''
    def __init__(self, data_dir, output_dir):
        super(KalmanFilterGeneric, self).\
              __init__(data_dir, output_dir, 1, 1)
        self.state = dict()  # objects being tracked, object id -> object state
        # initialize parameters for Kalman filtering
        dt = self.parameters['dt']
        pos_sigma = self.parameters['pos_sigma']
        a_sigma = self.parameters['a_sigma']
        eye2 = np.eye(2, dtype=float)
        zeros2 = np.zeros(shape=[2,2], dtype=float)
        # initialize F (evolution) and H (observation)
        self.F = np.vstack([
                            np.hstack([eye2, eye2*dt]),
                            np.hstack([zeros2, eye2])
                          ])
        self.H = np.hstack([eye2, zeros2])
        # initialize Q (evolution uncertainty) and R (observation uncertainty)
        G = np.vstack([eye2*dt*dt/2, eye2*dt])
        self.Q = a_sigma*a_sigma*matmul(G, np.transpose(G))
        self.R = pos_sigma*pos_sigma*eye2

    '''
    Read in observations for given frame, and initialize newly appeared
    objects.
    '''
    def UpdateTrackedObjects(self, frame):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Predict the state at next step, followed by reading in observations and
    improving the estimates.
    '''
    def Estimate(self, frame):
        # predcit state at next step, for each object
        for oid, state in self.state.items():
            xp, xcov = state.Predict(self.F, self.Q)
            state.SetPredictedState(xp)
            state.SetStateCov(xcov)
            state.ResetDone()  # need to estimate state in the next phase
        # read in observations, associate objects, initialize new objects,
        # remove objects that are no longer being tracked
        self.UpdateTrackedObjects(frame)
        # improve estimates, for each object
        for oid, state in self.state.items():
            if state.IsDone():  # skip new/marked objects
                continue
            xe, xcov = state.Update(state.GetObservation(), self.H, self.R)
            state.SetEstimatedState(xe)
            state.SetStateCov(xcov)
            state.MarkDone()


'''
Initializes state for an object that just appeared, using passed parameters.
Prioir information used for initializing object reflects how new objects are
created in SimpleRandomSimulator.
'''
def SimpleRandomInitializer(parameters, F, Q, H, R, pos):
    eye2 = np.eye(2, dtype=float)
    zeros2 = np.zeros(shape=[2,2], dtype=float)
    pos_sigma = parameters['pos_sigma']
    a_sigma = parameters['a_sigma']
    # initialize cov
    cov = np.vstack([
                     np.hstack([pos_sigma*pos_sigma*eye2, zeros2]),  # position
                     np.hstack([zeros2, a_sigma*a_sigma*eye2])       # velocity
                   ])
    # expected values after first step
    v_mean  = parameters['v_mean']  # v-magnitude
    delta   = v_mean * parameters['dt']  # dist from boundary edge
    mean    = np.array([pos[0]-delta, pos[1]-delta,
                        pos[0]-(1-delta), pos[1]-(1-delta)])
    # find which side this object is most likely to have come from,
    # and estimate initial velocity using that
    side = np.argmin(np.abs(mean))
    v = [0, 0]
    if side == 0:
        v[0] = v_mean
    elif side == 1:
        v[1] = v_mean
    elif side == 2:
        v[0] = -v_mean
    else:
        assert(side == 3)
        v[1] = -v_mean
    po = np.reshape(np.array(pos, dtype=float), [2,])
    xe = np.reshape(np.array(pos + v, dtype=float), [4,])
    return KalmanObjectState(x_obs=po, x_est=xe, x_pred=xe,
                             x_cov=cov, done=True)


'''
Basic Kalman filter.
This does not perform any object association between two steps.
It assumes taht the association is known, and just estimates the true state.
It initializes object state for newly appeared objects using the known
distribution for SimpleRandomSimulator.
'''
class KalmanFilterBasic(KalmanFilterGeneric):
    '''
    The init method takes in input data directory, output directory,
    and an initializer method for newly appeared objects.
    '''
    def __init__(self, data_dir, output_dir,
                 ObjectInitializer=SimpleRandomInitializer):
        super(KalmanFilterBasic, self).\
            __init__(data_dir, output_dir)
        self.InitializeObject = ObjectInitializer

    '''
    Read observations for given frame.
    If an object was present in the prevbious frame, update the observed
    position of the object.
    If this object appeared for the first time, initialize its estimated state,
    and mark it as done.
    If an object in self.state was not observed, that is, it was there in the
    previous frame, but disappeared, remove that object from self.state.
    '''
    def UpdateTrackedObjects(self, frame):
        file_name = os.path.join(self.data_dir, 'state_%08d.txt' % frame)
        if not os.path.isfile(file_name):
            print('Error, did not find state file for frame %d' % frame)
            exit(1)
        input_data = data_reader.ReadStateWithID(file_name)
        # stop tracking objects that are not observed
        tracking = self.state.keys()
        for oid in tracking:
            if oid not in input_data:
                del self.state[oid]
        # update state of objects that are being onserved
        for oid, obj_state in  input_data.items():
            # update observed position for objects being tracked 
            if oid in self.state:
                object_state = self.state[oid]
                object_state.SetObservation(np.reshape(
                    np.array(obj_state[2], dtype=float), [2,]))
            # initialize new objects
            else:
                object_state = self.InitializeObject(
                    self.parameters, self.F, self.Q,
                    self.H, self.R, obj_state[2])
                object_state.MarkDone()
                self.state[oid] = object_state
            self.state[oid].SetTrueState(
                np.reshape(np.array(obj_state[0], dtype=float), [2,]),
                np.reshape(np.array(obj_state[1], dtype=float), [2,]))


'''
Distance between observation and predicted state.
'''
def ObsTrackDist(state, obs_pos):
    pred_state = state.GetPredictedState()
    pred_pos = np.array(pred_state[0:2], dtype=float)
    return np.sqrt(np.sum(np.power(obs_pos - pred_pos, 2)))


'''
Distance between predicted state and closest boundary.
'''
def DistFromClosestBoundary(state):
    pred_state = state.GetPredictedState()
    pos = pred_state[0:2]
    xmin = min(pos[0], 1-pos[0])
    ymin = min(pos[1], 1-pos[1])
    return (max(min(xmin, ymin), 0))

'''
Multivariate normal.
'''
def NormalPDF(mean, covariance, x):
    cov_inv_x = matmul(np.linalg.inv(covariance), (x-mean))
    t = -0.5 * matmul(np.transpose(x-mean), cov_inv_x)
    d = np.linalg.det(2*math.pi*covariance)
    return math.exp(t)/math.sqrt(d)


'''
Discretize and precompute normal pdf.
'''
nside = 200
ndim  = 2*nside + 1
pdf_scale = 8.0/float(2*nside)
# go from -4*sigma to 4*sigma, with ndim points
computed_pdfs = dict()
def InitializeNormalPDF(name, covariance):
    if name in computed_pdfs.keys():
        return
    pdf = np.zeros(shape=[ndim, ndim])
    computed_pdfs[name] = pdf
    cov_det_rt = np.sqrt(np.linalg.det(covariance))
    A = covariance / cov_det_rt
    delta = (pdf_scale * matmul(A, np.ones(shape=[2,], dtype=float)))
    zv2 = np.zeros(shape=[2,], dtype=float)
    for i in range(ndim):
        for j in range(ndim):
            il, jl = i - nside, j - nside
            x = np.reshape(np.array([il*delta[0], jl*delta[1]]), [2,])
            computed_pdfs[name][i,j] = NormalPDF(zv2, covariance, x)
def PDF(name, x, y):
    i, j = int(round(x)), int(round(y))
    if i < 0 or i > ndim:
        return 0
    if j < 0 or j > ndim:
        return 0
    return computed_pdfs[name][i,j]

'''
Probability that an observation is from a track, without occlusion.
'''
def ObsTrackProb(state, obs_pos, covariance):
    pred_state = state.GetPredictedState()
    pred_pos = pred_state[0:2]
    pdf = NormalPDF(pred_pos, covariance, obs_pos)
    cov_det_rt = np.sqrt(np.linalg.det(covariance))
    A = covariance / cov_det_rt
    delta = (pdf_scale * matmul(A, np.ones(shape=[2,], dtype=float)))
    return pdf * delta[0] * delta[1]

'''
Probability that a track is new or left.
'''
def MatchOutsideProb(pos, covariance):
    cov_det_rt = np.sqrt(np.linalg.det(covariance))
    A = covariance / cov_det_rt
    delta = (pdf_scale * matmul(A, np.ones(shape=[2,], dtype=float)))
    prob = 0
    for i in range(ndim):
        for j in range(ndim):
            il, jl = i - nside, j - nside
            mpos = np.reshape(
                np.array([pos[0]+il*delta[0], pos[1]+jl*delta[1]]), [2,])
            if mpos[0] < 0 or mpos[0] > 1 or mpos[1] < 0 or mpos[1] > 1:
                prob += PDF('noise', i,j)
            else:
                continue
    return prob * delta[0] * delta[1]

'''
Returns true if a point seems occluded to a viewer.
'''
def IsOccluded(pos, visible_mask):
    if (pos[0] < 0 or pos[0] > 1 or pos[1] < 0 or pos[1] > 1):
        return False
    n = np.shape(visible_mask)[0]
    coord = np.round(pos*n)
    if coord[0] < 0 or coord[0] >= 500 or coord[1] < 0 or coord[1] >= 500:
        return False
    return (visible_mask[int(coord[0]), int(coord[1])] == 0)

'''
Probability that a track is occluded.
'''
def MatchOccludedProb(pos, covariance, visible_mask):
    cov_det_rt = np.sqrt(np.linalg.det(covariance))
    A = covariance / cov_det_rt
    delta = (pdf_scale * matmul(A, np.ones(shape=[2,], dtype=float)))
    prob = 0
    for i in range(ndim):
        for j in range(ndim):
            il, jl = i - nside, j - nside
            mpos = np.reshape(
                np.array([pos[0]+il*delta[0], pos[1]+jl*delta[1]]), [2,])
            if IsOccluded(mpos, visible_mask):
                prob += PDF('noise', i,j)
            else:
                continue
    return prob * delta[0] * delta[1]

'''
Probability that an observation matches a track, using prior distribution on
the track.
'''
def ObsTrackProbMAP(state, obs_pos, covariance):
    pred_state = state.GetPredictedState()
    pred_pos = pred_state[0:2]
    return NormalPDF(pred_pos, covariance, obs_pos)

'''
Probability that a track is new or left, using prior distribution on the track.
'''
def MatchOutsideProbMAP(pos, covariance, delta, num_pts):
    prob = 0
    side_pts = int((num_pts-1)/2)
    for i in range(num_pts):
        for j in range(num_pts):
            il, jl = i - side_pts, j - side_pts
            mpos = np.reshape(
                np.array([pos[0]+il*delta, pos[1]+jl*delta]), [2,])
            if mpos[0] < 0 or mpos[0] > 1 or mpos[1] < 0 or mpos[1] > 1:
                prob += NormalPDF(pos, covariance, mpos)
            else:
                continue
    return prob

'''
Probability that a track is occluded, using prior distribution on the track.
'''
def MatchOccludedProbMAP(pos, covariance, visible_mask, delta, num_pts):
    prob = 0
    side_pts = int((num_pts-1)/2)
    for i in range(num_pts):
        for j in range(num_pts):
            il, jl = i - side_pts, j - side_pts
            mpos = np.reshape(
                np.array([pos[0]+il*delta, pos[1]+jl*delta]), [2,])
            if IsOccluded(mpos, visible_mask):
                prob += NormalPDF(pos, covariance, mpos)
            else:
                continue
    return prob


'''
Brute-force search optimal match given observations, tracks, and match/no-match
errors.
All observations must match to a track or create a new track.
Penalty for unmatched tracks and observations are given by
unassigned_track_errirs abd unassigend_obs_errors.
'''
def SearchOptimalRecursive(unassigned_track_errors, unassigned_obs_errors,
                           gated_tracks, assigned_tracks, obs_id):
    num_tracks = len(unassigned_track_errors)
    num_obs = len(gated_tracks)
    gated = gated_tracks[obs_id]
    errors = dict()
    assignments = dict()
    # rec_loss, the loss if we do not match this observation to anything
    # (create a new track for unonccluded case)
    obs_unmatched_error = unassigned_obs_errors[obs_id]
    # case: this observation is unmatched
    # obs_unmatched_error loss for not matching the observation
    # (creating a new track)
    rec_error, rec_match = 0, dict()
    if obs_id+1 < num_obs:  # not the last observation
        rec_error, rec_match = SearchOptimalRecursive(
            unassigned_track_errors, unassigned_obs_errors,
            gated_tracks, assigned_tracks, obs_id+1
        )
    errors[num_tracks] = obs_unmatched_error + rec_error
    assignments[num_tracks] = rec_match
    # a penalty for each unassigned track in case this is the last
    # observation (last resursive call stack)
    if obs_id == num_obs-1:
        errors[num_tracks] += \
                sum([unassigned_track_errors[i] for i in range(num_tracks)
                     if not assigned_tracks[i]])
    # case: this observation is from one of existing tracks
    for tid, error in gated.items():
        # check that track is unassigned, and error is finite
        if not assigned_tracks[tid]:
            assigned_tracks[tid] = True
            rec_error, rec_match = 0, dict()
            if obs_id+1 < num_obs:  # not the last observation
                rec_error, rec_match = SearchOptimalRecursive(
                    unassigned_track_errors, unassigned_obs_errors,
                    gated_tracks, assigned_tracks, obs_id+1
                )
            # error for this assignment is [observation-track_pos] error plus
            errors[tid] = error + rec_error
            # a penalty for each unassigned track in case this is the last
            # observation (last resursive call stack)
            if obs_id == num_obs-1:
                errors[tid] += sum([unassigned_track_errors[i] for i in
                                range(num_tracks) if not assigned_tracks[i]])
            assigned_tracks[tid] = False
            assignments[tid] = rec_match
    # get assignment that gives smallest error
    min_error_tid = min(errors, key=errors.get)
    min_error = errors[min_error_tid]
    min_assignment = assignments[min_error_tid]
    min_assignment[obs_id] = dict()
    if min_error_tid == len(unassigned_track_errors):
        # create a new track for this observation
        min_assignment[obs_id]['new_track'] = True
    else:
        # minimum error is when this observation is from an existing track
        min_assignment[obs_id]['new_track'] = False
        min_assignment[obs_id]['track_id'] = min_error_tid
    return min_error, min_assignment


gate_factor = 5

'''
Get optimal match given observations and predicted tracks for one frame.
Does a brute-force search with gating to restrict observation-track match pairs
to a bounded region around each observation/track.
All observations must match to a track or create a new track. If a track is
unmatched, then there is a penalty, computed using NoObsTrackError.
'''
def SearchOptimalNearest(state_all, observations, parameters, visible_mask):
    pos_sigma = parameters['pos_sigma']
    v_mean = parameters['v_mean']
    a_sigma = parameters['a_sigma']
    dt = parameters['dt']
    sigma = a_sigma*dt*dt/2.0 + pos_sigma
    threshold = gate_factor*sigma
    num_tracks = len(state_all)
    num_obs = len(observations)
    # gated tracks with track-obs match score
    gated_tracks = dict()
    track_error = [list()]*num_tracks
    for obs_id, obs_pos in enumerate(observations):
        gated = dict()
        gated_tracks[obs_id] = gated
        for tid, t_state in state_all.items():
            error = ObsTrackDist(t_state, obs_pos)
            if error <= threshold:
                gated[tid] = error
                track_error[tid].append(error)
    # penalty for not assigning a track
    unassigned_track_errors = [float('inf')] * num_tracks
    for tid, t_state in state_all.items():
        t_error = DistFromClosestBoundary(t_state)
        # penalize for choosing to not associate a track with any observation
        if t_error < threshold:
            unassigned_track_errors[tid] = max(max(track_error[tid]), t_error)
    # penalty for creating a new track for an observation
    unassigned_obs_errors = [0] * num_obs
    # search optimal
    assigned_tracks = [False] * num_tracks
    error, match = SearchOptimalRecursive(
        unassigned_track_errors, unassigned_obs_errors,
        gated_tracks, assigned_tracks, 0)
    return match


'''
Get optimal match given observations and predicted tracks for one frame.
Does a brute-force search with gating to restrict observation-track match pairs
to a bounded region around each observation/track.
All observations must match to a track or create a new track. If a track is
unmatched, then there is a penalty, computed using NoObsTrackError.
'''
def SearchOptimalUnoccludedML(state_all, observations, parameters,
                              visible_mask):
    # parameters and probability distribution
    pos_sigma = parameters['pos_sigma']
    v_mean = parameters['v_mean']
    a_sigma = parameters['a_sigma']
    dt = parameters['dt']
    viewer_pos = parameters['viewer_pos']
    sigma = a_sigma*dt*dt/2.0 + pos_sigma
    threshold = gate_factor*sigma
    noise_covariance = np.eye(2, dtype=float) * sigma * sigma
    mean = np.zeros(shape=[2,], dtype = float)
    num_tracks = len(state_all)
    num_obs = len(observations)
    # precompute pdf
    InitializeNormalPDF('noise', noise_covariance)
    # gated tracks with track-obs match score
    gated_tracks = dict()
    for obs_id, obs_pos in enumerate(observations):
        gated = dict()
        gated_tracks[obs_id] = gated
        for tid, t_state in state_all.items():
            t_state = state_all[tid]
            dist = ObsTrackDist(t_state, obs_pos)
            if dist <= threshold:
                prob = ObsTrackProb(t_state, obs_pos, noise_covariance)
                gated[tid] = -math.log(prob)
    # penalty for not assigning a track
    unassigned_track_errors = [float('inf')] * num_tracks
    for tid, t_state in state_all.items():
        pred_state = t_state.GetPredictedState()
        pred_pos = pred_state[0:2]
        prob = MatchOutsideProb(pred_pos, noise_covariance)
        if prob != 0:
            unassigned_track_errors[tid] = -math.log(prob)
    # penalty for creating a new track for an obervation
    unassigned_obs_errors = [float('inf')] * num_obs
    for obs_id, obs_pos in enumerate(observations):
        prob = MatchOutsideProb(obs_pos, noise_covariance)
        if prob != 0:
            unassigned_obs_errors[obs_id] = -math.log(prob)
    # search optimal
    assigned_tracks = [False] * num_tracks
    error, match = SearchOptimalRecursive(
        unassigned_track_errors, unassigned_obs_errors,
        gated_tracks, assigned_tracks, 0)
    return match


'''
Get optimal match given observations and predicted tracks for one frame.
Does a brute-force search with gating to restrict observation-track match pairs
to a bounded region around each observation/track.
Penalizes unmatched observations and unmatched tracks.
'''
def SearchOptimalOccludedML(state_all, observations, parameters,
                            visible_mask):
    # parameters and probability distribution
    pos_sigma = parameters['pos_sigma']
    v_mean = parameters['v_mean']
    a_sigma = parameters['a_sigma']
    dt = parameters['dt']
    viewer_pos = parameters['viewer_pos']
    sigma = a_sigma*dt*dt/2.0 + pos_sigma
    threshold = gate_factor*sigma
    noise_covariance = np.eye(2, dtype=float) * sigma * sigma
    mean = np.zeros(shape=[2,], dtype = float)
    num_tracks = len(state_all)
    num_obs = len(observations)
    # precompute pdf
    InitializeNormalPDF('noise', noise_covariance)
    # gated tracks with track-obs match score
    gated_tracks = dict()
    for obs_id, obs_pos in enumerate(observations):
        gated = dict()
        gated_tracks[obs_id] = gated
        for tid, t_state in state_all.items():
            t_state = state_all[tid]
            dist = ObsTrackDist(t_state, obs_pos)
            if dist <= threshold:
                prob = ObsTrackProb(t_state, obs_pos, noise_covariance)
                gated[tid] = -math.log(prob)
    # penalty for not assigning a track
    # 2 cases here, one is track goes out, second is track is occluded
    unassigned_track_errors = [float('inf')] * num_tracks
    for tid, t_state in state_all.items():
        pred_state = t_state.GetPredictedState()
        pred_pos = pred_state[0:2]
        prob_outside  = MatchOutsideProb(pred_pos, noise_covariance)
        prob_unoccluded = MatchOccludedProb(pred_pos, noise_covariance, visible_mask)
        prob = max(prob_outside, prob_unoccluded)
        if prob != 0:
            unassigned_track_errors[tid] = -math.log(prob)
    # penalty for creating a new track for an obervation
    # this does not include occluded regions of previous frame,
    # to keep things simpler
    unassigned_obs_errors = [float('inf')] * num_obs
    for obs_id, obs_pos in enumerate(observations):
        prob = MatchOutsideProb(obs_pos, noise_covariance)
        if prob != 0:
            unassigned_obs_errors[obs_id] = -math.log(prob)
    # search optimal
    assigned_tracks = [False] * num_tracks
    error, match = SearchOptimalRecursive(
        unassigned_track_errors, unassigned_obs_errors,
        gated_tracks, assigned_tracks, 0)
    return match


'''
Get optimal match given observations and predicted tracks for one frame.
Does a brute-force search with gating to restrict observation-track match pairs
to a bounded region around each observation/track.
Penalizes unmatched observations and unmatched tracks, and uses prior
distribution on track estimates.
'''
def SearchOptimalOccludedMAP(state_all, observations, parameters,
                             visible_mask):
    # parameters and probability distribution
    pos_sigma = parameters['pos_sigma']
    v_mean = parameters['v_mean']
    a_sigma = parameters['a_sigma']
    dt = parameters['dt']
    viewer_pos = parameters['viewer_pos']
    sigma = a_sigma*dt*dt/2.0 + pos_sigma
    threshold = gate_factor*sigma
    noise_covariance = np.eye(2, dtype=float) * pos_sigma * pos_sigma
    mean = np.zeros(shape=[2,], dtype = float)
    num_tracks = len(state_all)
    num_obs = len(observations)
    per_sigma = 8
    delta = sigma/per_sigma
    num_pts = 4*per_sigma+1
    H = np.hstack([np.eye(2, dtype=float), np.zeros(shape=[2,2], dtype=float)])
    # precompute pdf
    InitializeNormalPDF('noise', noise_covariance)
    # gated tracks with track-obs match score
    gated_tracks = dict()
    for obs_id, obs_pos in enumerate(observations):
        gated = dict()
        gated_tracks[obs_id] = gated
        for tid, t_state in state_all.items():
            t_state = state_all[tid]
            dist = ObsTrackDist(t_state, obs_pos)
            if dist <= threshold:
                prob = ObsTrackProb(t_state, obs_pos, noise_covariance)
                gated[tid] = -math.log(prob)
    # penalty for not assigning a track
    # 2 cases here, one is track goes out, second is track is occluded
    unassigned_track_errors = [float('inf')] * num_tracks
    for tid, t_state in state_all.items():
        pred_state = t_state.GetPredictedState()
        pred_pos = pred_state[0:2]
        state_cov = matmul(H, matmul(t_state.GetStateCov(),
                                           np.transpose(H)))
        total_cov = state_cov + noise_covariance
        prob_outside  = MatchOutsideProbMAP(pred_pos, total_cov, delta, num_pts)
        prob_unoccluded = MatchOccludedProbMAP(pred_pos, total_cov,
                                               visible_mask, delta, num_pts)
        prob = max(prob_outside, prob_unoccluded)
        if prob != 0:
            unassigned_track_errors[tid] = -math.log(prob)
    # penalty for creating a new track for an obervation
    # this does not include occluded regions of previous frame,
    # to keep things simpler
    unassigned_obs_errors = [float('inf')] * num_obs
    for obs_id, obs_pos in enumerate(observations):
        prob = MatchOutsideProb(obs_pos, noise_covariance)
        if prob != 0:
            unassigned_obs_errors[obs_id] = -math.log(prob)
    # search optimal
    assigned_tracks = [False] * num_tracks
    error, match = SearchOptimalRecursive(
        unassigned_track_errors, unassigned_obs_errors,
        gated_tracks, assigned_tracks, 0)
    return match


'''
All readings are available, but there is no information on how many tracks
there really are.
Error = distance between track and observed distance, this corresponds to
maximum likelihood when all observations are available (no occlusion).
Association is done using brute force search, for Kalman filter based predictor.
'''
class KalmanFilterWithAssociation(KalmanFilterGeneric):
    '''
    The init method takes in input data directory, output directory,
    and an initializer method for newly appeared objects.
    '''
    def __init__(self, data_dir, output_dir, occluded=False,
                 OptimalMatch=SearchOptimalNearest,
                 ObjectInitializer=SimpleRandomInitializer
                ):
        super(KalmanFilterWithAssociation, self).\
            __init__(data_dir, output_dir)
        self.InitializeObject = ObjectInitializer
        self.OptimalMatch     = OptimalMatch
        self.occluded         = occluded
        # delete tracks not matched 4 consecutive times for occluded case
        # if the track is not near the center, for some definition of center
        self.not_matched_limit_boundary = 2
        self.not_matched_limit_center = 3
        self.eps = 4 * self.parameters['pos_sigma']

    def DeleteTrack(self, state, visible_mask):
        pos = state.GetPredictedState()
        if IsOccluded(pos, visible_mask):
            return False
        if (pos[0] > self.eps and pos[0] < 1-self.eps and
            pos[1] > self.eps and pos[1] < 1-self.eps):
            return (state.NumTimesUnmatched() >= self.not_matched_limit_center)
        return (state.NumTimesUnmatched() >= self.not_matched_limit_boundary)


    '''
    Read observations for given frame, and match observations to tracks.
    Initialize new tracks.
    '''
    def UpdateTrackedObjects(self, frame):
        file_name = os.path.join(self.data_dir, 'state_%08d.txt' % frame)
        if not os.path.isfile(file_name):
            print('Error, did not find state file for frame %d' % frame)
            exit(1)
        input_data = data_reader.ReadStateShuffled(file_name)
        # occluded case, keep only the visible data
        if self.occluded:
            input_data = [d for d in input_data if d[3]]
        # read observations
        obs_data = [np.reshape(np.array(obs_pos, dtype=float), [2,])
                    for _, _, obs_pos, _ in input_data]
        visible_mask = utils.ReadImage(
            os.path.join(self.data_dir, 'visible_%08d.png' % frame))
        # compute best match
        match = self.OptimalMatch(
            self.state, obs_data, self.parameters, visible_mask)
        state = dict()
        num_objects = len(obs_data)
        # update state using computed match
        tracks_matched = set()
        for oid, m in match.items():
            if not m['new_track']:
                tid = m['track_id']
                object_state = self.state[tid]
                object_state.SetOldTrack()
                object_state.RecordMatched()
                obs_pos = input_data[oid][2]
                object_state.SetObservation(
                    np.reshape(np.array(obs_pos, dtype=float), [2,]))
                state[oid] = object_state
                tracks_matched.add(tid)
            else:
                object_state = self.InitializeObject(
                    self.parameters, self.F, self.Q,
                    self.H, self.R, input_data[oid][2])
                object_state.MarkDone()
                object_state.SetNewTrack()
                state[oid] = object_state
            state[oid].SetTrueState(
                np.reshape(np.array(input_data[oid][0], dtype=float), [2,]),
                np.reshape(np.array(input_data[oid][1], dtype=float), [2,]))
        # if track unmatched for some number of observations, stop tracking
        # keep tracks that have been matched recently enough number of times,
        # and are inside the domain
        if self.occluded:
            num_tracks = len(self.state)
            for tid in range(num_tracks):
                if tid not in tracks_matched:
                    object_state = self.state[tid]
                    if self.DeleteTrack(object_state, visible_mask):
                        continue
                    object_state.SetOldTrack()
                    object_state.RecordUnmatched()
                    # don't try to update estimate as there is no observation
                    object_state.MarkDone()
                    state[len(state)] = object_state
        # set state
        self.state = state
