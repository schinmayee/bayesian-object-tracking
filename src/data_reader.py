'''
Reads actual positions and velocity of objects, from file file_name, and
returns a dictionary of object id -> (position, velocity).
'''
def ReadActualState(file_name):
    lines = open(file_name, 'r')
    num_objects = int(file_data[0])
    # number of lines is number of objects, plus 1st line -- header,
    # plus may be an additional empty line at the end
    assert(len(lines) == num_objects+1 or len(lines) == num_objects+2)
    # dictionary of object id -> true state
    true_state = dict()
    for o in range(1,num_objects+1):
        obj_data = lines[o].split()
        obj_id = int(obj_data[0])
        obj_pos = [float(obj_data[i]) for i in [1,2]]
        obj_vel = [float(obj_data[i]) for i in [3,4]]
        true_state[o] = (obj_pos, obj_vel)
    return true_state

'''
Reads observed positions of objects, from file file_name, and returns a
shuffled list of observed positions.
The position in the list does not have any implication.
'''
#TODO: update this to return only those objects that are visible
def ReadObservedStateShuffled(file_name):
    lines = open(file_name, 'r')
    num_objects = int(file_data[0])
    # number of lines is number of objects, plus 1st line -- header,
    # plus may be an additional empty line at the end
    assert(len(lines) == num_objects+1 or len(lines) == num_objects+2)
    # dictionary of object id -> true state
    obs_state = dict()
    for o in range(1,num_objects+1):
        obj_data = lines[o].split()
        obs_pos = [float(obj_data[i]) for i in [5,6]]
        obs_state[o] = obs_pos
    random.shuffle(obs_state)
    return obs_state
