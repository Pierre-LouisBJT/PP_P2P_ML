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
    
    for neighbor in np.nonzero(W[agent])[0].tolist():
        learningPart += W[agent][neighbor] * np.array(model[neighbor][-1]) / W[agent][agent]


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

#Loader of ml-100k dataset
def load_ml100k(path): #path (str) to folder, ends with '/'
    #get number of agents
    with open(path + 'u.info', 'r') as f:
        lines = f.readlines()
        n = int(lines[0].split()[0])
        f.close()
    print('number of agents : {}'.format(n))

    #load the data 
    with open(path + 'u.data', 'r') as f:
        lines = f.readlines()
        f.close()
    rawdata = []
    for line in lines:
        splited_line = line.split('	')
        int_line = []
        for x in splited_line: #convert str to int
            int_line.append(int(x))
        rawdata.append(int_line)
    #Warning : at this stage, the object rawdata isn't the final data

    #create the agents_data_idx 
    agents_data_idx = []
    for i in range(0,n):
        agents_data_idx.append([])
    for idx in range(0,len(rawdata)):
        agents_data_idx[rawdata[idx][0] - 1] = idx

    #create the final data object
    #extract infos from u.item
    with open(path + 'u.item', 'r', encoding = "ISO-8859-1") as f:
        lines = f.readlines()
        f.close()
    items = []
    for line in lines:
        rawitem = line.split('|')
        rawitem_refined = rawitem[5:len(rawitem)-1]
        rawitem_refined.append(rawitem[-1][0])
        item = []
        for x in rawitem_refined:
            item.append(int(x))
        items.append(item)

    data = []
    for x in rawdata:
        item_id = x[1]
        rating = x[2]
        data.append([items[item_id - 1], rating]) #item_idx = item_id - 1
    print('Dataset is loaded!')
    return data, agents_data_idx