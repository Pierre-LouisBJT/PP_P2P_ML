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