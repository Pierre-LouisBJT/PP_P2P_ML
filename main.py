"""
This is the main script, which us functions from modules.

Theta (global param) is a list of dictionnaries.
"""
from modules import * #TODO syntax

def train(data, agents_matrix, agents_data_idx, privacy, nb_steps):
    """
    random initialization
    for each step in range nb_steps :
        for each agent:
            if agent wakes up 
                update local theta_i (4)
                compute new local param
                broadcast step
                calculate time before next wake up (random.poisson(lam=1.0, size=None))
    """
    
    return False

def evaluate(data, model): #makes predictions using a model
    return False


print('Hello world')