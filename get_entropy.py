'''
Joshua Meyer

This script takes a corpus and outputs the entropy of the most frequent words
given their POS tags. 

INPUT: list of (word,POS) tuples

OUTPUT: list of words and entropies in descending order

USAGE: $ python3 get_entropy.py
'''

from collections import Counter
import re
import numpy as np


def get_tokenPOS_tuples(fileName):
    tokenPOSlist = []
    f = open(fileName)

    for line in f:
        line = line.rstrip()        
        for word_POS in line.split(' '):
            try:
                tokenPOSlist.append(tuple(word_POS.split('_')))
            except:
                pass
    return tokenPOSlist


def compute_entropy(tokenPOSlist):
    '''
    Given a list of (word,POS) tuples, find the posterior probability, aka
    p(POS|word)
    '''
    # get frequencies of words, POS, and word+POS
    POSlist=[]
    tokenList=[]
    try:
        for token,POS in tokenPOSlist:
            POSlist.append(POS)
            tokenList.append(token)
    except:
        pass
    POSFreqs = Counter(POSlist)
    tokenFreqs = Counter(tokenList)
    tokenPOSFreqs = Counter(tokenPOSlist)

    freqDict={}
    for key,value in POSFreqs.items():
        freqDict[key]=value
    for key,value in tokenFreqs.items():
        freqDict[key]=value
    for key,value in tokenPOSFreqs.items():
        freqDict[key]=value

    # take each word and POS, retrieve joint frequencies
    wordStats=[]
    for word in set(tokenList):
        POSfreqs=[]
        for POS in set(POSlist):
            try:
                freq = freqDict[(word,POS)]
            except KeyError:
                freq = 0.01
            POSfreqs.append((POS,freq))
        freqs = [freq for POS,freq in POSfreqs]
        # compute all the entropies!
        entropy = -sum([(freq/sum(freqs))*np.log2(freq/sum(freqs))
                              for freq in freqs])
        wordStats.append([word,entropy,POSfreqs])
    return wordStats


if __name__ == "__main__":
            
    # open up the corpus
    fileName = input("Enter filepath here: ")

    tokenPOSList = get_tokenPOS_tuples(fileName)

    # # throw out all words which aren't VB, NN, or JJ
    # nounADJverb=[]
    # for tokenPOS in tokenPOSList:
    #     try:
    #         if tokenPOS[1] in ["VB","NN","JJ"]:
    #             nounADJverb.append((tokenPOS[0],tokenPOS[1]))
    #     except:
    #         pass
    #
    # wordStats = compute_entropy(nounADJverb)
    wordStats = compute_entropy(tokenPOSList)

    orderedList = sorted(wordStats, key=lambda x: x[1], reverse=True)

    outFile = open('zeroentropy.txt', 'w')
    for word in orderedList:
        print(word, end='\n', file=outFile)

