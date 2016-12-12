#!/usr/bin/env python

import argparse, os
import math
import numpy as np
import pickle
import random
import utils

script_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(script_dir, '..', 'data/random')

parser = argparse.ArgumentParser("Generate object, options: ")

parser.add_argument('--frames', dest='frames', type=int, required=True,
    help='Number of frames to simulate')
parser.add_argument('--log', dest='log_freq', type=int, default=1,
    help='Logging frequency')
parser.add_argument('--output', dest='output_dir', type=str, default=output_dir,
    help='Output directory')
parser.add_argument('--max_objects', dest='max_objects', type=int, default=10,
    help='Max number of objects in a frame')

args = parser.parse_args()
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
assert(os.path.isdir(args.output_dir))

occluded_color = 150
noise_color = 75

'''
Generic object in simulation.
All objects have a position and velocity.
Child classes may have additional attributes.
'''
class SimObject(object):
    def __init__(self, pos, vel):
        self.pos = pos
        self.pos_obs = pos
        self.vel = vel

    '''
    Set object id
    '''
    def SetId(self, obj_id):
        self.id = obj_id

    '''
    Add noise to pos, for observed position.
    '''
    def AddNoiseToPos(self, noise):
        self.pos_obs = self.pos + noise

    '''
    Return true if object is in frame, even partially.
    Objects that are not in frame will be removed, that is, not tracked further,
    in the simulator.
    '''
    def IsInFrame(self):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Update distance and angle information with respect to viewer.
    '''
    def UpdateDistanceAndAngle(self, viewer_pos):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Compute if object is occuded or not, and mark the object accordingly.
    '''
    def MarkUnoccluded(self, viewer_pos, objects):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Return true if object is not occluded/
    '''
    def IsUnoccluded(self):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Set pixels that are occupied by the object, in im_arr.
    '''
    def SetImage(self, im_arr, n):
        raise NotImplemented('Call not implemented. Override in child class!')

def IsUnoccluded(viewer_pos, dist, angle, half_angle, objects):
    angle_min = angle - half_angle
    angle_max = angle + half_angle
    for o in objects:
        if not o.IsUnoccluded():  # ignore occluded objects
            continue
        if o.dist_obs < dist:
            o_min = o.angle_obs - o.half_angle_obs
            o_max = o.angle_obs + o.half_angle_obs
            if angle_max > o_min and angle_max < o_max:
                angle_max = min(angle_max, o_min)
            if angle_min > o_min and angle_min < o_max:
                angle_min = max(angle_min, o_max)
    return (angle_min < angle_max)


'''
Circular object, inherits from generic SimObject.
'''
class CircObject(SimObject):
    def __init__(self, center, vel, r, eps=0.001):
        super(CircObject, self).__init__(center, vel)
        self.radius = r
        self.eps = eps
        self.angle = None  # angle that object's center forms with y axis
        self.half_angle = None  # half angle formed by the object at viewer
        self.dist = None  # distance from viewer
        # observed angles and distance is what sensor sees
        # should exclude truly occluded objects which the sensor does not see
        self.angle_obs = None  # angle that object's center forms
        self.half_angle_obs = None  # half angle formed by the object
        self.dist_obs = None  # distance from viewer
        self.unoccluded = False

    def IsInFrame(self):
        return (self.pos[0] + self.radius >= -self.eps and
                self.pos[0] - self.radius <= 1 + self.eps and
                self.pos[1] + self.radius >= -self.eps and
                self.pos[1] - self.radius <= 1 + self.eps)

    def UpdateDistanceAndAngle(self, viewer_pos):
        rel_pos = viewer_pos - self.pos  # angle from y axis
        self.angle = math.atan2(rel_pos[1], rel_pos[0])
        self.dist = np.sqrt(np.sum(np.power(rel_pos, 2)))
        if self.dist < self.radius:
            self.half_angle = math.pi/2
        else:
            self.half_angle = math.asin(self.radius/self.dist)
        rel_pos_obs = viewer_pos - self.pos_obs  # angle from y axis
        self.angle_obs = math.atan2(rel_pos_obs[1], rel_pos_obs[0])
        self.dist_obs = np.sqrt(np.sum(np.power(rel_pos_obs, 2)))
        if self.dist_obs < self.radius:
            self.half_angle_obs = math.pi/2
        else:
            self.half_angle_obs = math.asin(self.radius/self.dist_obs)


    def MarkUnoccluded(self, viewer_pos, objects):
        angle_min = self.angle - self.half_angle
        angle_max = self.angle + self.half_angle
        for o in objects:
            if self == o:
                continue
            if o.dist < self.dist: 
                o_min, o_max = o.angle-o.half_angle, o.angle+o.half_angle
                if angle_max > o_min and angle_max < o_max:
                    angle_max = min(angle_max, o_min)
                if angle_min > o_min and angle_min < o_max:
                    angle_min = max(angle_min, o_max)
        self.unoccluded = (angle_min < angle_max)

    def IsUnoccluded(self):
        return self.unoccluded

    def SetImage(self, im_arr, n):
        d_cell = 1.0/n
        min_pt = np.floor((self.pos - self.radius)/d_cell)
        max_pt = np.ceil((self.pos + self.radius)/d_cell) + 1
        x_min, y_min = max(int(min_pt[0]), 0), max(int(min_pt[1]), 0)
        x_max, y_max = min(int(max_pt[0]), n), min(int(max_pt[1]), n)
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                pt = np.array([i, j])*d_cell + d_cell/2  # cell center
                dist = np.sqrt(np.sum(np.power(pt - self.pos, 2)))  # distance from circ center
                if dist <= self.radius:
                    if self.unoccluded:
                        im_arr[i,j] = 255
                    else:
                        im_arr[i,j] = occluded_color
        min_pt = np.floor((self.pos_obs - self.radius)/d_cell)
        max_pt = np.ceil((self.pos_obs + self.radius)/d_cell) + 1
        x_min, y_min = max(int(min_pt[0]), 0), max(int(min_pt[1]), 0)
        x_max, y_max = min(int(max_pt[0]), n), min(int(max_pt[1]), n)
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                pt = np.array([i, j])*d_cell + d_cell/2  # cell center
                dist = np.sqrt(np.sum(np.power(pt - self.pos_obs, 2)))  # distance from circ center
                if dist <= self.radius and im_arr[i,j] == 0:
                    im_arr[i,j] = noise_color

'''
Generic simulator, evolves objects with a fixed time step, using simple
kinematic equations.
* The simulator does not handle collisions right now.
* The velocity at every time step is updated using acceleration from
GetAcceleration.
* Progress and images are logged every log_freq frames.
* Simulator runs for frames number of frames.
'''
class Simulator(object):
    def __init__(self, dt, max_objects, output_dir):
        self.dt = dt
        self.max_objects = max_objects
        self.output_dir = output_dir
        self.objects = list()
        self.object_ids_used = 0
        self.viewer_pos = np.array([1.0, 0.5], dtype=float)

    '''
    Update dictionary parameters with parameter key,values
    '''
    def GetParameters(self, parameters):
        parameters['dt'] = self.dt
        parameters['viewer_pos'] = self.viewer_pos
        return parameters

    '''
    Create and return a new object with some initial position and velocity.
    This is called every time the number of objects is less than max_objects.
    If a simulator decides to not create an object at some time step,
    it can return None.
    '''
    def CreateNewObject(self):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Return an acceleration for a given object.
    '''
    def GetAcceleration(self, o):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Update observed position for object.
    '''
    def UpdateObservedPosition(self, o):
        raise NotImplemented('Call not implemented. Override in child class!')

    def SaveVisibleMask(self, file_name):
        raise NotImplemented('Call not implemented. Override in child class!')

    '''
    Save position and velocity of all objects in a text file.
    Data is saved as a list of oject id, true position, true velocity and
    observed position.
    The first line, the header, gives the number of objects, and each
    following line corresponds to one object.
    '''
    # TODO: change this to a csv format to make it more manageable as more
    # columns are added
    def SaveState(self, f):
        file_name = os.path.join(self.output_dir, 'state_%08d.txt'%f)
        with open(file_name, 'w') as data:
            data.write('%d\n' % len(self.objects))  # number of objects
            for o in self.objects:
                data.write('%i ' % o.id)  # object id
                o.pos.tofile(data, sep=' ')  # pos
                data.write(' ')
                o.vel.tofile(data, sep=' ')  # vel
                data.write(' ')
                o.pos_obs.tofile(data, sep=' ')  # pos_obs
                data.write(' ')
                data.write(str(o.IsUnoccluded()))
                data.write('\n')

    '''
    Save image with all current objects, for visualization.
    '''
    def SaveImageAndSensor(self, f):
        n = 500
        im_arr = np.zeros(shape=[n,n], dtype=np.uint8)
        for o in self.objects:
            o.SetImage(im_arr, n)

        viewer_coord = self.viewer_pos * n
        vx, vy = int(viewer_coord[0]), int(viewer_coord[1])
        for i in range(-6, 7):
            for j in range(-6, 7):
                x, y = vx+i, vy+j
                if x >= 0 and x < n and y >= 0 and y < n:
                    im_arr[x,y] = occluded_color
        for i in range(-3, 4):
            for j in range(-3, 4):
                x, y = vx+i, vy+j
                if x >= 0 and x < n and y >= 0 and y < n:
                    im_arr[x,y] = noise_color

        utils.SaveImage(im_arr, os.path.join(self.output_dir, 'state_%08d.png'%f))

        self.SaveVisibleMask(os.path.join(self.output_dir, 'visible_%08d.png'%f))


    '''
    Run the simulator for frames number of frames.
    Images and progress is logged every log_freq frames.
    The simulator does not handle collisions right now.
    '''
    def Run(self, frames, log_freq):
        parameters = dict()
        self.GetParameters(parameters)
        parameter_file_name = os.path.join(self.output_dir, 'parameters')
        with open(parameter_file_name, 'w') as parameter_file:
            pickle.dump(parameters, parameter_file)

        for f in range(frames):
            # log status
            if f%log_freq == 0:
                print('Simulating frame %d ...' %f)

            # create new object
            if len(self.objects) < self.max_objects:
                obj = self.CreateNewObject()
                if obj is not None:
                    self.object_ids_used += 1
                    obj.SetId(self.object_ids_used)
                    self.objects.append(obj)

            keep_objects = list()
            for o in self.objects:
                # evolve object
                a = self.GetAcceleration(o)
                o.pos = o.pos + o.vel*self.dt + 0.5*a*self.dt*self.dt
                o.vel = o.vel + a*self.dt

                # remove objects that are out of frame
                if o.IsInFrame():
                    self.UpdateObservedPosition(o)
                    keep_objects.append(o)

            self.objects = keep_objects

            for o in self.objects:
                o.UpdateDistanceAndAngle(self.viewer_pos)

            # unoccluded objects
            for o in self.objects:
                o.MarkUnoccluded(self.viewer_pos, self.objects)

            # save all object data
            self.SaveState(f)
            if f%log_freq == 0:
                self.SaveImageAndSensor(f)

'''
Simple random circular object simulator, inherits from generic Simulator.
It creates circular objects of fixed radius, and fixed initial velocity.
The objects accelerate with an acceleration value drawn from a zero mean
gaussian distribution.
'''
class SimpleRandomSimulator(Simulator):
    def __init__(self, dt, max_objects, output_dir):
        super(SimpleRandomSimulator, self).__init__(dt, max_objects, output_dir)
        self.v_mean = 1.0/(50.0*dt)
        self.a_mean = 0
        self.a_sigma = self.v_mean
        self.prob = max_objects*self.v_mean*dt
        self.radius = 0.05
        self.pos_sigma = self.radius/2

    def GetParameters(self, parameters):
        super(SimpleRandomSimulator, self).GetParameters(parameters)
        parameters['v_mean'] = self.v_mean
        parameters['a_mean'] = self.a_mean
        parameters['a_sigma'] = self.a_sigma
        parameters['pos_sigma'] = self.pos_sigma

    def CreateNewObject(self):
        create_toss = random.random()
        if len(self.objects) < self.max_objects/2 or create_toss < self.prob:
            point = random.random()
            side = random.randint(0,3)
            pos, vel = None, None
            if side == 0:
                pos = np.array([-self.radius, point])
                vel = np.array([self.v_mean, 0])
            elif side == 1:
                pos = np.array([point, -self.radius])
                vel = np.array([0, self.v_mean])
            elif side == 2:
                pos = np.array([1+self.radius, point])
                vel = np.array([-self.v_mean, 0])
            else:
                assert(side == 3)
                pos = np.array([point, 1+self.radius])
                vel = np.array([0, -self.v_mean])
            return CircObject(pos, vel, self.radius)
        else:
            return None

    def GetAcceleration(self, o):
        ax = random.gauss(self.a_mean, self.a_sigma)
        ay = random.gauss(self.a_mean, self.a_sigma)
        return np.array([ax, ay])

    def UpdateObservedPosition(self, o):
        px = random.gauss(0, self.pos_sigma)
        py = random.gauss(0, self.pos_sigma)
        o.AddNoiseToPos(np.array([px, py]))

    def SaveVisibleMask(self, file_name):
        n = 500
        delta = 1/float(n)
        im_mask = np.zeros(shape=[n,n], dtype=np.uint8)
        for i in range(n):
            for j in range(n):
                pos = np.array([i*delta, j*delta], dtype=float)
                rel_pos = self.viewer_pos - pos
                dist = np.sqrt(np.sum(np.power(rel_pos, 2)))
                angle = math.atan2(rel_pos[1], rel_pos[0])
                if dist < self.radius:
                    half_angle = math.pi/2
                else:
                    half_angle = math.asin(self.radius/dist)
                if IsUnoccluded(
                    self.viewer_pos, dist, angle, half_angle, self.objects):
                    im_mask[i,j] = 255
        utils.SaveImage(im_mask, file_name)


if __name__ == '__main__':
    random.seed(0)

    simulator = SimpleRandomSimulator(0.1, args.max_objects, args.output_dir)
    simulator.Run(args.frames, args.log_freq)
