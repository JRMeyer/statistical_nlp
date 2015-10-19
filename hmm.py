'''
Joshua Meyer

Simulating a Hidden Markov Model

USAGE: $ python q2.py
'''
import numpy as np
import random

class HMM:
    def __init__(self):
        # number of states in model
        self.N = 3
        # number of elements in emissions alphabet
        self.M = 3
        # array of initial state probabilities
        self.PI = np.array([0.3, 0.5, 0.2],
                      dtype=float)
        # matrix of state transition probabilities
        self.A = np.matrix([[0.2, 0.6, 0.2],
                           [0.3, 0.3, 0.4],
                           [0.1, 0.8, 0.1]],
                          dtype=float)
        # matrix of emission probabilities
        self.B = np.matrix([[0.3, 0.1, 0.6],
                           [0.5, 0.3, 0.2],
                           [0.1, 0.7, 0.2]],
                          dtype=float)
        # look-up tables for emissions and states
        self.S = ['1','2','3']
        self.K = ['a','b','c']

    
    def generate_string(self):
        pass
    
    def evaluate_string(self, O):
        # create matrix to hold local probailities
        # one row for each state
        # one column for each element in observance,
        # plus one column for initial state probs
        self.Trellis = np.zeros((len(self.S),
                                 len(O)+1),
                                dtype=float)
        
        # chose an initial state and set to 1
        initialState = random.randint(0,2)
        self.Trellis[initialState,0] = 1
        lastBest = initialState
        
        trellisCol=1
        for emission in O:
            colMax=0
            emissionIndex = self.K.index(emission)
            for state in range(len(self.S)):
                transProb = self.A[lastBest,state]
                emissionProb = self.B[state,emissionIndex]
                combinedProb = transProb*emissionProb
                
                self.Trellis[state,trellisCol] = combinedProb
                
                if combinedProb > colMax:
                    combinedProb = colMax
                else:
                    pass
            lastBest = colMax
            trellisCol+=1

        # \Sigma_{X_1 ... X_{T+1}} \pi_{X_i} \prod_{t+1}^{T} a_{X_t X_{t+1}} \cdot b_{X_t,X_t+1} o_t
        print(np.sum(np.prod(self.Trellis, axis=1)))
        
if __name__ == "__main__":
    hmm = HMM()
    _string = input("enter string")
    hmm.evaluate_string(_string)


# how to generate things (from tutorial)
# (1) choose initial state from PI
# (2) set t = 1
# (3) chose emission from initial state (B) 
# (4) transition to new state (A)
# (5) increment t and go to (3) 
