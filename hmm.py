'''
Joshua Meyer

Simulating a Hidden Markov Model

USAGE: $ python q2.py
'''
import numpy as np

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
        self.emissions = ['a','b','c']
        self.states = ['1','2','3']
    
    def generate_string(self):
        pass
    
    def evaluate_string(self, O):
        # create matrix to hold local probailities,
        # with one row for each state
        # one column for each element in observance sequence,
        # plus one column for initial state probs
        self.Trellis = np.zeros((len(self.states),
                                 len(O)+1),
                                dtype=float)
        print(self.Trellis)


if __name__ == "__main__":
    hmm = HMM()
    hmm.evaluate_string('abbbb')


# how to generate things (from tutorial)
# (1) choose initial state from PI
# (2) set t = 1
# (3) chose emission from initial state (B) 
# (4) transition to new state (A)
# (5) increment t and go to (3) 
