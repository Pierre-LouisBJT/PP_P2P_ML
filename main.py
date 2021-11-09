"""
This is the main script, which us functions from modules.

Theta (global param) is a list of dictionnaries.
"""
import random
import numpy as np

from modules import * #TODO syntax

def train(data, matrix, agents_data_idx, privacy, max_steps, mu, alpha, L): #d is the dim of x
    """
    random initialization
    for each step in range nb_steps :
        for each agent:
            if agent wakes up 
                update local theta_i (4)
                broadcast step
                calculate time before next wake up (random.poisson(lam=1.0, size=None))
    """
    n = len(matrix) #matrix is a list of lists
    d = len(data[0])
    model = []
    clocks = [] #n times (int) wher the agent will wake up
    neighbors = [] #list of the indexs (int) of the neighbors for each agent
    C = [] #list of n float, confidence coeff for each agent
    #random init of the model
    for i in range(0, n):
        submodel = []
        for j in range(0, n):
            theta = []
            for k in range (0,d):
                theta.append(2*random.random() - 1) #TODO change init
            submodel.append(theta)
        model.append(submodel)
    
    #set the first wakeup times
    for i in range(0, n):
        clocks.append(np.random.poisson(lam=1.0, size=None))

    #calculate neighbors
    for agent in range(0, n):
        neighbors.append(getNeighbors(matrix, agent))

    #calculate confidence coeff
    for agent in range(0, n):
        C.append(len(agents_data_idx[agent]))
    m_max = max(C)
    for agent in range(0, n):
        C[agent] = C[agent]/m_max

    #run of the algo for each step
    for step in range(0, max_steps):
        for agent in range (0, n):
            if step == clocks[agent] : #agent wakes up
                #update local theta_i
                model = updateStep(model, matrix, C, mu, alpha, L) #TODO missing args?
                #broadcast step
                model = broadcastStep(model, neighbors, agent)
                #calculate time before next wake up
                clocks[agent] = step + np.random.poisson(lam=1.0, size=None)

    return model

def evaluate(data, model): #makes predictions using a model
    return False


print('Hello world')