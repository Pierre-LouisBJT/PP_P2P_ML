"""
This is where we put functions
"""

import numpy as np #TODO: don't import twice

def W(data): # construct the adjency matrix of the graph, using method from paper for ml-100k
    return False

#Update step
def objectiveFun():
    return False

def objectiveFunGrad():
    return False

def computeGrad():
    return False

def loss(theta, x, y): #(theta.T * x - y)**2, quadratic loss by default, where theta is current local for agent
    return (np.dot(theta, x) - y)**2

def localLossFun(model, agents_data_idx, lambd, agent):
    theta = model[agent][-1] #current local theta

    localLoss = 0

    for j in agents_data_idx[agent]:
        localLoss += loss(theta, data[j][0], data[j][1])

    localLoss /= len(agents_data_idx[agent])

    localLoss += lambd[agent] * theta**2
    return localLoss

def updateStep(model, W, agent, agents_data_idx, C, mu, alpha, lambd):
    theta = model[agent][-1]
    learningPart = 0
    
    theta_new = (1 - alpha[agent]) * theta + alpha[agent] * learningPart
    model[agent].append(theta)
    return model

#broadcast step
def broadcastStep(model, neighbors, agent):
    if len(neighbors)>0:
        for neighbor in neighbors:
            model[neighbor] = model[agent]
    return model

def getNeighbors(W, agent): #W, int
    n = len(W)
    neighbors = []
    for i in range(0, n):
        if i != agent:
            if W[agent][i] > 0:
                neighbors.append(i)
    return neighbors

class agent : #idx, W, model?

    def computeLocalParam(self, data, model, steps): #update local theta
        loss = 0
        return loss