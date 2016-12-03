#!/usr/bin/env python

import random

'''
Reads file file_name, and returns a
dictionary of object id -> (true pos, true vel, observed pos).
'''
def ReadStateWithID(file_name):
    with open(file_name, 'r') as file_data:
        lines = file_data.readlines()
        num_objects = int(lines[0])
        # number of lines is number of objects, plus 1st line -- header,
        # plus may be an additional empty line at the end
        assert(len(lines) == num_objects+1 or len(lines) == num_objects+2)
        # dictionary of object id -> true state
        obs_state = dict()
        for o in range(1,num_objects+1):
            obj_data = lines[o].split()
            oid = int(obj_data[0])
            obj_pos = [float(obj_data[i]) for i in [1,2]]
            obj_vel = [float(obj_data[i]) for i in [3,4]]
            obs_pos = [float(obj_data[i]) for i in [5,6]]
            obs_state[oid] = (obj_pos, obj_vel, obs_pos)
        return obs_state

'''
Reads file file_name, and returns a
shuffled list of (true pos, true vel, observed pos).
The position in the list does not imply anything about how the object is
associated with objects from another frame.
Multiple calls to this with the same file_name can return a different
permutation of object positions.
'''
def ReadStateShuffled(file_name):
    with open(file_name, 'r') as file_data:
        lines = file_data.readlines()
        num_objects = int(lines[0])
        # number of lines is number of objects, plus 1st line -- header,
        # plus may be an additional empty line at the end
        assert(len(lines) == num_objects+1 or len(lines) == num_objects+2)
        # list of observed state
        obs_state = list()
        for o in range(1,num_objects+1):
            obj_data = lines[o].split()
            obj_pos = [float(obj_data[i]) for i in [1,2]]
            obj_vel = [float(obj_data[i]) for i in [3,4]]
            obs_pos = [float(obj_data[i]) for i in [5,6]]
            obs_state.append((obj_pos, obj_vel, obs_pos))
        random.shuffle(obs_state)
        return obs_state
