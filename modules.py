"""
This is where we put functions
"""

def construct_matrix(data): # construct the adjency matrix of the graph, using method from paper for ml-100k
    return False

#Update step
def computeGrad():
    return False

def updateStep(model, matrix, C, mu, alpha, L):
    return model

#broadcast step
def broadcastStep(model, neighbors, agent):
    if len(neighbors)>0:
        for neighbor in neighbors:
            model[neighbor] = model[agent]
    return model

def getNeighbors(matrix, agent): #matrix, int
    n = len(matrix)
    neighbors = []
    for i in range(0, n):
        if i != agent:
            if matrix[agent][i] > 0:
                neighbors.append(i)
    return neighbors

class agent : #idx, matrix, model?

    def computeLocalParam(self, data, model, steps): #update local theta
        loss = 0
        return loss