#!/usr/bin/python3
"""
@author: Mayank Manjrekar

The hard-core birth-death process is a model where balls of radius one
arrive at random, and are accepted if thare no overlapping balls
present in the system. Every point has an exponential lifetime.

In this SimPy simulation, we run a bi-simulation for this
process, i.e., a simultaneous simulation of two processes, starting
from different initial conditions, but driven by the same
arrivals. This kind of simulation will be useful in understanding the
mixing times of the process.

The particles only present in the process-0 are called zombies (BLUE), and
those present only in process-1 are called anti-zombies (GREEN). Those
that are present in both are called regular (RED).
"""

import matplotlib.animation as manimation
import simpy
import numpy as np
import matplotlib.pyplot as plt
exprand = np.random.exponential
rand = np.random.random



DIM = 2
SIZE = 20   # Size of the domain is (SIZE)^DIM
LAMBDA = 2   # The arrival rate
GEN_TIME_CONST = 1 / (LAMBDA * SIZE**2)   # Total arrival rate
LIFETIME = 0.125    # The average time a particle stays
UNTIL_TIME = 180    # The run-time of the simulation
NUMZOMBIES = 5      # Maximum number of zombies initially present in
                    # the system
ALIVE = 1
DEAD = 0


TORUS = []
for i in range(DIM):
    temp = [[k+[j]] for k in TORUS for j in range(-1,2)]
    TORUS = temp
TORUS = SIZE * np.array(TORUS)

CAPTURE_MOVIE = True
UPDATE_PLOT = False

# Particle class holds the information of a particle
class Particle():
    def __init__(self, iden, x_p, b_p, l_p, s_p):
        self.id = iden
        self.location = x_p
        self.birth_time = b_p
        self.lifetime = l_p
        self.state = s_p     # State = Vector of ALIVE or DEAD in
                             # process_{0,1}
        env.process(self.run())
        
    def run(self): # The lifetime process of a particle. It waits till
                   # the lifetime runs out.
        yield env.timeout(self.lifetime)
        hash_loc = np.floor(self.location)
        self.state = [DEAD, DEAD]
        t_hash_loc = tuple(hash_loc)
        Plist[t_hash_loc].remove(self) 
        print("{0:3f} : Particle {1} departs".format(env.now, self.id))
        

#def hashloc(loc):
#    return np.floor(loc) % SIZE

def adjSquares(ind, hashed_loc):
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
            rt = rt + adjSquares(ind - 1, temp)
        return rt


def distance(loc1, loc2):
    return min(np.linalg.norm((TORUS + loc1 - loc2), axis=1))


# Find the state of a particle after arrival
def stateAtArrival(loc): 
    state = [ALIVE, ALIVE]
    hash_loc = np.floor(loc)
    ind = len(loc) - 1
    for p_bucket in adjSquares(ind, hash_loc):
        if tuple(p_bucket) in Plist:
            for pt in Plist[tuple(p_bucket)]:
                if distance(pt.location, loc) < 1:
                    if pt.state[0] == ALIVE:
                        state[0] = DEAD
                    if pt.state[1] == ALIVE:
                        state[1] = DEAD
                    if state[0] == state[1] == DEAD:
                        return state
    return state

# The particle generator process
def particleGenerator():
    iden = 1
    # First we generate the Zombies at time 0
    for i in range(NUMZOMBIES):
        loc = np.array([rand(), rand()]) * SIZE
        hash_loc = np.floor(loc) % SIZE
        state = stateAtArrival(loc)
        state[0] = DEAD
        if state[1] == ALIVE:
            print("{0:3f} : Particle {1} arrives as a Zombie".format(env.now, iden, state))
            new_pt = Particle(iden, loc, env.now, exprand(1 / LIFETIME), state)
            t_hash_loc = tuple(hash_loc)
            if t_hash_loc in Plist:
                Plist[t_hash_loc].append(new_pt)
            else:
                Plist[t_hash_loc] = [new_pt]
            iden += 1
    while True:
        yield env.timeout(exprand(GEN_TIME_CONST)) # Wait till new arrival
        loc = np.array([rand(), rand()]) * SIZE
        hash_loc = np.floor(loc) % SIZE
        state = stateAtArrival(loc)
        if state[0] == ALIVE or state[1] == ALIVE:
            print("{0:3f} : Particle {1} arrives with state {2}, end_sim = {3}".format(env.now, iden, state, end_sim))
            new_pt = Particle(iden, loc, env.now, exprand(1 / LIFETIME), state)
            t_hash_loc = tuple(hash_loc)
            if t_hash_loc in Plist:
                Plist[t_hash_loc].append(new_pt)
            else:
                Plist[t_hash_loc] = [new_pt]
        iden += 1
        if end_sim: # Another way to safely exit the simulation
            env.exit()
        


def colorpt(state):
    if state[0]==state[1]==ALIVE:
        return [1,0,0,1] #RED
    elif state[0]==ALIVE: 
        return [0,1,0,1] #BLUE
    else:
        return [0,0,1,1] #GREEN


def update_plot():
    plt.cla()
    global end_sim
    end_sim = True
    for bucket in Plist.values():
        for pt in bucket:
            ax1.add_artist(plt.Circle(pt.location, 0.5,alpha = 0.3, color = colorpt(pt.state)))   
            if pt.state[0] != pt.state[1] and end_sim:
                end_sim = False
    plt.axis((0, SIZE, 0, SIZE))
    if UPDATE_PLOT:
        plt.draw()
        plt.pause(0.0001)
    

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
    env.process(particleGenerator()) # Start Particle generator
    env.process(writer_grabber())    # Video capture process
    env.run(until = UNTIL_TIME)


# Setting up the SimPy simulation
env = simpy.Environment()
Plist = dict()
fig1, ax1 = plt.subplots()
plt.axis('equal')
end_sim = False
main()
