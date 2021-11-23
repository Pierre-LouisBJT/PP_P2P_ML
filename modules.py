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

def loss(theta, x, y): # float, (theta.T * x - y)**2, quadratic loss by default, where theta is current local for agent
    return (np.dot(theta, x) - y)**2

def lossGrad(theta, x, y): # list of n float, 2(theta.T * x - y)x, grad of quadratic loss by default, where theta is current local for agent
    return 2 * (np.dot(theta, x) - y) * x

def localLossFun(model, agents_data_idx, lambd, agent): #float
    theta = model[agent][-1] #current local theta

    localLoss = 0

    for j in agents_data_idx[agent]:
        localLoss += loss(theta, data[j][0], data[j][1])

    localLoss /= len(agents_data_idx[agent])

    localLoss += lambd[agent] * np.linalg.norm(theta, ord=2)**2
    return localLoss

def localLossFunGrad(model, agents_data_idx, lambd, agent): #list of n float
    theta = model[agent][-1] #current local theta

    localLossGrad = 0

    for j in agents_data_idx[agent]:
        localLossGrad += lossGrad(theta, data[j][0], data[j][1])

    localLossGrad /= len(agents_data_idx[agent])

    localLossGrad += 2 * lambd[agent] * theta
    return localLossGrad

def updateStep(model, W, agent, agents_data_idx, C, mu, alpha, lambd):
    theta = model[agent][-1]
    learningPart = 0
    
    for neighbor in agents_data_idx[agent]:
        learningPart += W[agent][neighbor] * model[neighbor][-1]/ W[agent][agent]


    learningPart -= mu * C[agent] * localLossFunGrad(model, agents_data_idx, lambd, agent)

    theta_new = (1 - alpha[agent]) * theta + alpha[agent] * learningPart
    model[agent].append(theta_new)
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