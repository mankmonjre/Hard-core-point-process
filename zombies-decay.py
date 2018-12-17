# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 11:12:55 2016

@author: mayankm

This is a simulation for a coupling of a system where the hard-core 
balls arrive at random, and are accepted if there are no competing balls
are present in a radius 1 from it. Every point accepted has an exponential
lifetime.
"""
import matplotlib.animation as manimation
import simpy
import numpy as np
import matplotlib.pyplot as plt

SIZE = 20
LAMBDA = 2
GEN_TIME_CONST = 1 / (LAMBDA * SIZE**2)
LIFETIME = 0.125
UNTIL_TIME = 180
NUMZOMBIES = 5



exprand = np.random.exponential

rand = np.random.random

ALIVE = 1
DEAD = 0
TORUS = SIZE * np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])

CAPTURE_MOVIE = True
UPDATE_PLOT = True

class Point():
    def __init__(self, iden, x_p, b_p, l_p, s_p):
        self.id = iden
        self.location = x_p
        self.birth_time = b_p
        self.lifetime = l_p
        self.state = s_p
        env.process(self.run())
        
    def run(self):
        yield env.timeout(self.lifetime)
        hashed_loc = np.floor(self.location)
        self.state = [DEAD, DEAD]
        t_hashed_loc = tuple(hashed_loc)
        points[t_hashed_loc].remove(self) 
        print("{0:3f} : Point {1} departs".format(env.now, self.id))
        

def hashloc(loc):
    return np.floor(loc) % SIZE

def grid(ind, hashed_loc):
    if ind == 0:
        rt = []
        for i in range(-1,2):
            temp = list(hashed_loc)
            temp[0] += i
            rt.append(temp)
        return rt
    else:
        rt = []
        for i in range(-1,2):
            temp = list(hashed_loc)
            temp[ind] += i
            rt = rt + grid(ind - 1, temp)
        return rt


def distance(loc1, loc2):
    return min(np.linalg.norm((TORUS + loc1 - loc2), axis=1))


def state_at_arrival(loc):
    state = [ALIVE, ALIVE]
    hashed_loc = hashloc(loc)
    ind = len(loc) - 1
    for p_bucket in grid(ind, hashed_loc):
        if tuple(p_bucket) in points:
            for pt in points[tuple(p_bucket)]:
                if distance(pt.location, loc) < 1:
                    if pt.state[0] == ALIVE:
                        state[0] = DEAD
                    if pt.state[1] == ALIVE:
                        state[1] = DEAD
                    if state[0] == state[1] == DEAD:
                        return state
    return state


def point_gen_proc():
    iden = 1
    for i in range(NUMZOMBIES):
        loc = np.array([rand(), rand()]) * SIZE
        hashed_loc = hashloc(loc)
        state = state_at_arrival(loc)
        state[0] = DEAD
        if state[1] == ALIVE:
            print("{0:3f} : Point {1} arrives as a Zombie".format(env.now, iden, state))
            new_pt = Point(iden, loc, env.now, exprand(1 / LIFETIME), state)
            t_hashed_loc = tuple(hashed_loc)
            if t_hashed_loc in points:
                points[t_hashed_loc].append(new_pt)
            else:
                points[t_hashed_loc] = [new_pt]
            iden += 1
    while True:
        yield env.timeout(exprand(GEN_TIME_CONST))
        loc = np.array([rand(), rand()]) * SIZE
        hashed_loc = np.floor(loc)
        state = state_at_arrival(loc)
        if state[0] == ALIVE or state[1] == ALIVE:
            print("{0:3f} : Point {1} arrives with state {2} end_sim = {3}".format(env.now, iden, state, end_sim))
            new_pt = Point(iden, loc, env.now, exprand(1 / LIFETIME), state)
            t_hashed_loc = tuple(hashed_loc)
            if t_hashed_loc in points:
                points[t_hashed_loc].append(new_pt)
            else:
                points[t_hashed_loc] = [new_pt]
        iden += 1
        if end_sim:
            env.exit()
        


def colorpt(state):
    if state[0]==state[1]==ALIVE:
        return [1,0,0,1]
    elif state[0]==ALIVE:
        return [0,1,0,1]
    else:
        return [0,0,1,1]


def update_plot():
    #colors = []
    plt.cla()
    global end_sim
    end_sim = True
    for bucket in points.values():
        for pt in bucket:
            ax1.add_artist(plt.Circle(pt.location, 0.5,alpha = 0.3, color = colorpt(pt.state)))   
            if pt.state[0] != pt.state[1] and end_sim:
                end_sim = False
    plt.axis((0, SIZE, 0, SIZE))
    if UPDATE_PLOT:
        plt.draw()
        plt.pause(0.0001)

    
    #plt.draw()
    #plt.pause(0.00001)
            
    

def writer_grabber():
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)
    with writer.saving(fig1, "simvideo_lam{0}_lif{1}_s{2}.mp4".format(LAMBDA, LIFETIME, SIZE), 200):
        while env.now < UNTIL_TIME - 0.2:
            yield env.timeout(0.065)
            update_plot()
            if CAPTURE_MOVIE:
                writer.grab_frame()
            if end_sim:
                env.exit()
            




def main():
    np.random.seed(0)
    env.process(point_gen_proc())
    env.process(writer_grabber())
    env.run(until = UNTIL_TIME)



env = simpy.Environment()
points = dict()
fig1, ax1 = plt.subplots()
plt.axis('equal')
#plt.show()
end_sim = False
#plt.axes().set_aspect('equal', 'datalim')


main()
