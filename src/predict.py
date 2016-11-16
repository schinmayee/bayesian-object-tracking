#!/usr/bin/env python

import numpy as np
import pickle
import data_reader

class Predictor(object):
    def __init__(self, data_dir, output_dir, predict_steps=1, window=1):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.predict_steps = predict_steps
        self.window = window
        parameter_file_name = os.path.join(data_dir, 'parameters')
        with open(parameter_file_name) as parameter_file:
            self.parameters = pickle.load(parameter_file)
        print('Parameters :\n', self.parameters)

    def PredictAndObserve(self, frame):
        raise NotImplemented('Call')

    def SaveError(self, frame):
        raise NotImplemented('Call')

    def Run(self, frames, log_freq):
        for f in range(self.window, self.frames):
            self.PredictAndObserve(f)
            if f%log_freq == 0:
                self.SaveError(f)

class SimpleRandomPredictor(object):
    def __init__(self, data_dir):
        super(SimpleRandomPredictor, self).\
              __init__(data_dir, output_dir, 1, 1)
        dt = self.parameters.dt
        pos_sigma = self.parameters.pos_sigma
        a_sigma = self.parameters.a_sigma
        eye2 = np.eye(2, dtype=float)
        zeros2 = np.zeros(shape=[2,2], dtype=float)
        # initialize F and H
        self.F = np.vstack([
                            np.hstack([eye2, eye2*dt]),
                            np.hstack([zeros2, eye2])
                          ])
        self.H = np.hstack([np.eye(2),
                            np.zeros(shape=[2,2]])
        # initialize Q and R
        G = np.vstack([eye2*dt*dt/2, eye2*dt])
        self.Q = a_sigma*a_sigma*np.matmul(G, np.transpose(G))
        self.R = pos_sigma*pos_sigma*eye2
        # initialize P for first data point
        self.P = np.vstack([
                            np.hstack([pos_sigma*pos_sigma*eye2, zeros2]),
                            np.hstack([zeros2, a_sigma*a_sigma*eye2])
                          ])
        # read observed position of visible objects in the zero-th frame
        file_name = os.path.join(self.data_dir, 'state_%08d.txt'%0)
        pos_obs = data_reader.ReadObservedStateShuffled(file_name)
        # estimated object velocity in the zero-th frame
        vel_est = []
        # expected values after first step
        v_mean  = parameters.v_mean  # v-magnitude
        delta   = v_mean * parameters.dt  # dist from boundary edge
        mean    = np.array([delta[0], delta[1], 1-delta[0], 1-delta[1]])
        for p in self.pos_obs:
            # find which side this object is most likely to have come from
            side = np.argmin(np.abs(p-mean))
            v = np.zeros(np.shape([2,]))
            if side == 0:
                v[0] = v_mean
            elif side == 1:
                v[1] = v_mean
            elif side == 2:
                v[0] = -v_mean
            else:
                assert(side == 3)
                v[1] = -v_mean
            vel_est.append(v)
        self.state = np.array(pos_obs + vel_est, dtype=float)

    def PredictAndObserve(self, frame):
        # predcition
        x_pred = np.matmul(F, self.state)
        P_pred = np.matmul(np.matmul(F, self.P), np.transpose(F)) + Q
        file_name = os.path.join(self.data_dir, 'state_%08d.txt'%frame)
        pos_obs = data_reader.ReadObservedStateShuffled(file_name)
        state_obs = np.array(pos_obs+[0, 0], dtype=float)
        y = state_obs - np.matmul(H, x_pred)
        S = np.matmul(np.matmul(self.H, P_pred), np.transpose(H)) + self.R
        Sinv = np.linalg.inv(S)
        K = np.matmul(np.matmul(P_pred, np.transpose(self.H)), Sinv)
        self.state = x_pred + np.matmul(K, y)
        KH = np.matmul(K, H)
        self.P = np.matmul(np.eye(np.shape(KH)[0]) - KH, self.P)
