#!/usr/bin/env python

import pickle
import os
import math
import numpy as np
from scipy.stats import multivariate_normal as mvnorm
import data_reader
import utils

est_color_new = 25
est_color_old = 50


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

    '''
    Make predictions for next step, and read in data and refine the prediction.
    '''
    def Estimate(self, frame):
        raise NotImplemented('Call')

    '''
    Compute error in prediction.
    '''
    def ComputeError(self):
        raise NotImplemented('Call')

    '''
    Compute error in prediction.
    '''
    def SaveEstimateAsImage(self, frame):
        raise NotImplemented('Call')

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
        print(self.pos_error)
        print(self.vel_error)

'''
Object state, to use for Kalman filtering.
'''
class KalmanObjectState(object):
    zv2 = np.zeros(shape=[2], dtype=float)
    zv4 = np.zeros(shape=[4], dtype=float)
    eye4 = np.eye(4, dtype=float)

    def  __init__(self, x_obs=zv2, x_est=zv4,
                  x_pred=zv4, x_cov=eye4,
                  x_true=None, v_true=None, done=False):
        self.x_obs = x_obs    # observed position
        self.x_est = x_est    # estimated [pos:vel]
        self.x_pred = x_pred  # predicted [pos:vel]
        self.x_cov = x_cov    # state covariance (predicted/estimated)
        self.done = done
        self.x_true = x_true  # true position
        self.v_true = v_true  # true velocity
        self.new_track = True

    def SetTrueState(self, x_true, v_true):
        self.x_true = x_true
        self.v_true = v_true

    def GetTrueState(self):
        return (self.x_true, self.v_true)

    def SetObservation(self, x_obs):
        self.x_obs = x_obs

    def GetObservation(self):
        return self.x_obs

    def SetPredictedState(self, x_pred):
        self.x_pred = x_pred

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
        xp = np.matmul(F, self.GetEstimatedState())
        xcov = np.matmul(np.matmul(F, self.GetStateCov()), np.transpose(F)) + Q
        return xp, xcov

    def Update(self, obs, H, R):
        xp = self.GetPredictedState()
        xcov = self.GetStateCov()
        y = obs - np.matmul(H, xp)
        S = np.matmul(np.matmul(H, xcov), np.transpose(H)) + R
        Sinv = np.linalg.inv(S)
        K = np.matmul(np.matmul(xcov, np.transpose(H)), Sinv)
        xe = xp + np.matmul(K, y)
        KH = np.matmul(K, H)
        xcov = np.matmul(np.eye(np.shape(KH)[0]) - KH, self.GetStateCov())
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
        self.Q = a_sigma*a_sigma*np.matmul(G, np.transpose(G))
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
    po = np.array(pos, dtype=float)
    xe = np.array(pos + v, dtype=float)
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
                object_state.SetObservation(np.array(obj_state[2],
                                                     dtype=float))
            # initialize new objects
            else:
                object_state = self.InitializeObject(
                    self.parameters, self.F, self.Q,
                    self.H, self.R, obj_state[2])
                object_state.MarkDone()
                self.state[oid] = object_state
            self.state[oid].SetTrueState(np.array(obj_state[0], dtype=float),
                                         np.array(obj_state[1], dtype=float))

    '''
    Compute error.
    '''
    def ComputeError(self):
        frame_pos_error = 0
        frame_vel_error = 0
        for oid, state in self.state.items():
            est = state.GetEstimatedState()
            xe, ve = est[0:2], est[2:4]
            xt, vt = state.GetTrueState()
            frame_pos_error += np.sqrt(np.sum(np.power(xe-xt, 2)))
            frame_vel_error += np.sqrt(np.sum(np.power(ve-vt, 2)))
        num_objects = len(self.state)
        self.pos_error.append(frame_pos_error/num_objects)
        self.vel_error.append(frame_vel_error/num_objects)

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


def ObsTrackDist(state, obs_pos):
    pred_state = state.GetPredictedState()
    pred_pos = np.array(pred_state[0:2], dtype=float)
    return np.sqrt(np.sum(np.power(obs_pos - pred_pos, 2)))


def DistFromClosestBoundary(state):
    pred_state = state.GetPredictedState()
    pos = pred_state[0:2]
    xmin = min(pos[0], 1-pos[0])
    ymin = min(pos[1], 1-pos[1])
    return (max(min(xmin, ymin), 0))

def ObsTrackProb(state, obs_pos, covariance):
    pred_state = state.GetPredictedState()[0:2]
    pos = pred_state[0:2]
    return mvnorm.pdf(pred_state, mean=pos, cov=covariance)


'''
Brute-force search optimal match given observations, tracks, and match/no-match
errors.
All observations must match to a track or create a new track. If a track is
unmatched, then there is a penalty given by no-match error unassigned_errors.
'''
def SearchOptimalRecursive(unassigned_errors, gated_tracks,
                           assigned_tracks, obs_id):
    num_tracks = len(unassigned_errors)
    num_obs = len(gated_tracks)
    gated = gated_tracks[obs_id]
    errors = dict()
    assignments = dict()
    # case: this observation defines a new track
    rec_error, rec_match = 0, dict()
    if obs_id+1 < num_obs:  # not the last observation
        rec_error, rec_match = SearchOptimalRecursive(
            unassigned_errors, gated_tracks, assigned_tracks, obs_id+1
        )
    errors[num_tracks] = rec_error
    assignments[num_tracks] = rec_match
    if obs_id == num_obs-1:
        errors[num_tracks] += \
                sum([unassigned_errors[i] for i in range(num_tracks) if not
                     assigned_tracks[i]])
    # case: this observation is from one of existing tracks
    for tid, error in gated.items():
        # check that track is unassigned, and error is finite
        if not assigned_tracks[tid]:
            assigned_tracks[tid] = True
            rec_error, rec_match = 0, dict()
            if obs_id+1 < num_obs:  # not the last observation
                rec_error, rec_match = SearchOptimalRecursive(
                    unassigned_errors, gated_tracks, assigned_tracks, obs_id+1
                )
            # error for this assignment is [observation-track_pos] error plus
            # a penalty for each unassigned track in case this is the 0th
            # observation (top of resursive call stack)
            errors[tid] = error + rec_error
            if obs_id == num_obs-1:
                errors[tid] += sum([unassigned_errors[i] for i in
                                range(num_tracks) if not assigned_tracks[i]])
            assigned_tracks[tid] = False
            assignments[tid] = rec_match
    # get assignment that gives smallest error
    min_error_tid = min(errors, key=errors.get)
    min_error = errors[min_error_tid]
    min_assignment = assignments[min_error_tid]
    min_assignment[obs_id] = dict()
    if min_error_tid == len(unassigned_errors):
        # create a new track for this observation
        min_assignment[obs_id]['new_track'] = True
    else:
        # minimum error is when this observation is from an existing track
        min_assignment[obs_id]['new_track'] = False
        min_assignment[obs_id]['track_id'] = min_error_tid
    return min_error, min_assignment


'''
Get optimal match given observations and predicted tracks for one frame.
Does a brute-force search with gating to restrict observation-track match pairs
to a bounded region around each observation/track.
All observations must match to a track or create a new track. If a track is
unmatched, then there is a penalty, computed using NoObsTrackError.
Tracks that are outside the frame domain are not penalized.
'''
def SearchOptimalNearest(state_all, observations, parameters, visible_mask):
    v_mean = parameters['v_mean']
    a_sigma = parameters['a_sigma']
    dt = parameters['dt']
    threshold = 8*a_sigma*dt*dt
    num_tracks = len(state_all)
    gated_tracks = dict()
    track_error = [list()]*num_tracks
    for obs_id, obs_pos in enumerate(observations):
        gated = dict()
        gated_tracks[obs_id] = gated
        for tid, _ in enumerate(state_all):
            t_state = state_all[tid]
            error = ObsTrackDist(t_state, obs_pos)
            if error <= 8*threshold:
                gated[tid] = error
                track_error[tid].append(error)
    assigned_tracks = [False] * num_tracks
    unassigned_errors = [float('inf')] * num_tracks
    for tid, _ in enumerate(state_all):
        t_state = state_all[tid]
        t_error = DistFromClosestBoundary(t_state)
        # penalize for choosing to not associate a track with any observation
        if t_error < threshold:
            unassigned_errors[tid] = max(max(track_error[tid]), t_error)
    error, match = SearchOptimalRecursive(
        unassigned_errors, gated_tracks, assigned_tracks, 0)
    return match, unassigned_errors


def NotVisible(obs_pos, visible_mask):
    if (obs_pos[0] < 0 or obs_pos[0] > 1 or
        obs_pos[1] < 0 or obs_pos[1] > 1):
        return True
    n = np.shape(visible_mask)[0]
    coord = np.round(obs_pos/n)
    return (visible_mask[int(coord[0]), int(coord[1])] == 0)


'''
Get optimal match given observations and predicted tracks for one frame.
Does a brute-force search with gating to restrict observation-track match pairs
to a bounded region around each observation/track.
All observations must match to a track or create a new track. If a track is
unmatched, then there is a penalty, computed using NoObsTrackError.
Tracks that are outside the frame domain are not penalized.
'''
def SearchOptimalMostLikely(state_all, observations, parameters, visible_mask):
    pos_sigma = parameters['pos_sigma']
    v_mean = parameters['v_mean']
    a_sigma = parameters['a_sigma']
    dt = parameters['dt']
    viewer_pos = parameters['viewer_pos']
    threshold = max(2*a_sigma*dt*dt, v_mean*dt*math.sqrt(2))
    delta = 0.5/float(np.shape(visible_mask)[0])
    covariance = np.eye(2, dtype=float) * (a_sigma*a_sigma +
                                           pos_sigma*pos_sigma)
    gated_tracks = dict()
    for obs_id, obs_pos in enumerate(observations):
        gated = dict()
        gated_tracks[obs_id] = gated
        for tid, _ in enumerate(state_all):
            t_state = state_all[tid]
            dist = ObsTrackDist(t_state, obs_pos)
            if dist <= threshold:
                gated[tid] = -math.log(ObsTrackProb(
                    t_state, obs_pos, covariance))
    num_tracks = len(state_all)
    assigned_tracks = [False] * num_tracks
    unassigned_errors = [float('inf')] * num_tracks
    for tid, _ in enumerate(state_all):
        t_state = state_all[tid]
        prob = list()
        n = int(math.ceil(threshold/delta))
        for i in range(-n, n+1):
            for j in range(-n, n+1):
                if i*i + j*j <= n*n:
                    obs_pos = (t_state.GetPredictedState()[0:2] +
                               [float(i)*delta, float(j)*delta])
                    if NotVisible(obs_pos, visible_mask):
                        prob.append(ObsTrackProb(t_state, obs_pos, covariance))
        if len(prob) != 0:
            unassigned_errors[tid] = -math.log(max(prob))
    error, match = SearchOptimalRecursive(
        unassigned_errors, gated_tracks, assigned_tracks, 0)
    return match, unassigned_errors


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
    def __init__(self, data_dir, output_dir, unoccluded_only=False,
                 ObsTrackError=ObsTrackDist,
                 NoObsTrackError=DistFromClosestBoundary,
                 OptimalMatch=SearchOptimalNearest,
                 ObjectInitializer=SimpleRandomInitializer):
        super(KalmanFilterWithAssociation, self).\
            __init__(data_dir, output_dir)
        self.ObsTrackError    = ObsTrackDist
        self.NoObsTrackError  = DistFromClosestBoundary
        self.OptimalMatch     = OptimalMatch
        self.InitializeObject = ObjectInitializer
        self.unoccluded_only  = unoccluded_only

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
        if self.unoccluded_only:
            obs_data = [np.array(obs_pos, dtype=float)
                        for _, _, obs_pos, visible in input_data if visible]
        else:
            obs_data = [np.array(obs_pos, dtype=float) for _, _, obs_pos, _ in
                        input_data]
        visible_mask = utils.ReadImage(
            os.path.join(self.data_dir, 'visible_%08d.png' % frame))
        match, unassigned_errors = self.OptimalMatch(
            self.state, obs_data, self.parameters, visible_mask)
        state = dict()
        num_objects = len(obs_data)
        tracks_assigned = list()
        for oid, m in match.items():
            if not m['new_track']:
                tid = m['track_id']
                object_state = self.state[tid]
                object_state.SetOldTrack()
                obs_pos = input_data[oid][2]
                object_state.SetObservation(np.array(obs_pos, dtype=float))
                state[oid] = object_state
                tracks_assigned.append(tid)
            else:
                object_state = self.InitializeObject(
                    self.parameters, self.F, self.Q,
                    self.H, self.R, input_data[oid][2])
                object_state.MarkDone()
                object_state.SetNewTrack()
                state[oid] = object_state
            state[oid].SetTrueState(np.array(input_data[oid][0], dtype=float),
                                    np.array(input_data[oid][1], dtype=float))
        for tid, _ in enumerate(self.state):
            t_state = self.state[tid]
            if tid not in tracks_assigned:
                pos = t_state.GetPredictedState()[0:2]
                if pos[0] < 0 or pos[0] > 1 or pos[1] < 0 or pos[1] > 1:
                    continue
                if unassigned_errors[tid] == float('inf'):
                    continue
                t_state.MarkDone()
                t_state.SetOldTrack()
                state[num_objects] = t_state
                num_objects += 1

        self.state = state


    '''
    Compute error.
    '''
    def ComputeError(self):
        frame_pos_error = 0
        frame_vel_error = 0
        for oid, state in self.state.items():
            est = state.GetEstimatedState()
            xe, ve = est[0:2], est[2:4]
            xt, vt = state.GetTrueState()
            frame_pos_error += np.sqrt(np.sum(np.power(xe-xt, 2)))
            frame_vel_error += np.sqrt(np.sum(np.power(ve-vt, 2)))
        num_objects = len(self.state)
        self.pos_error.append(frame_pos_error/num_objects)
        self.vel_error.append(frame_vel_error/num_objects)

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
