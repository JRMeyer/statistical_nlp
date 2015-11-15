'''
Joshua Meyer

INPUT: list of (word,POS) tuples

OUTPUT: list of words and entropies in descending order

USAGE: $ python3 get_entropy.py
'''

from collections import Counter
from operator import itemgetter
import re
import numpy as np


def get_tokenPOS_tuples(lines):
    tokenPOSlist = []
    for line in lines:
        line = line.lower()
        line = line.rstrip()        
        for word_POS in line.split(' '):
            try:
                tokenPOSlist.append(tuple(word_POS.split('_')))
            except:
                pass
    return tokenPOSlist


def split_tokenPOS_tags(tokenPOSlist):
    POSlist=[]
    tokenList=[]
    try:
        for token,POS in tokenPOSlist:
            POSlist.append(POS)
            tokenList.append(token)
    except:
        pass
    return POSlist, tokenList


def get_freqDist_from_list(tokenPOSlist):
    '''
    Given  a list of (token,POS) tuples, create a freqDist which has
    word-POS pairs as keys, and the values are the respective frequencies
    '''
    tokenPOSFreqs = Counter(tokenPOSlist)
    
    freqDict={}
    for key,value in tokenPOSFreqs.items():
        freqDict[key]=value
        
    return freqDict

def get_naivePOS_dict(freqDist, POSlist, tokenList):
    wordStats={}
    for word in set(tokenList):
        POSfreqs=[]
        for POS in set(POSlist):
            try:
                freq = freqDict[(word,POS)]
            except KeyError:
                freq = 0.01
            POSfreqs.append((POS,freq))
        wordStats[word] = POSfreqs
        
    naivePOSdict={}
    for key,value in wordStats.items():
        naivePOSdict[key] = max(value,key=itemgetter(1))[0]
    return naivePOSdict


if __name__ == "__main__":
    
    # SPLIT INTO TRAINING AND TESTING CORPUS
    fileName = input("Enter filepath here: ")
    f = open(fileName)
    lines=[]
    for line in f:
        line = line.lower()
        line = line.rstrip()  
        lines.append(line)
    trainLines = lines[:int(round((len(lines)*.01),0))]
    testLines = lines[int(round((len(lines)*.01),0)):]

          
    # TRAINING
    tokenPOSlist = get_tokenPOS_tuples(trainLines)
    POSlist, tokenList = split_tokenPOS_tags(tokenPOSlist)
    defaultTag = max(set(POSlist), key=POSlist.count)
    freqDict = get_freqDist_from_list(tokenPOSlist)
    naivePOSdict = get_naivePOS_dict(freqDict, POSlist, tokenList)

    del tokenPOSlist,POSlist,tokenList,freqDict
    
    # TESTING
    wordError=0
    wordTotal=0
    sentenceError=0
    sentenceTotal=0
    for line in testLines:
        curSentError=0
        sentenceTotal+=1
        for tokenPOS in line.split(' '):
            wordTotal+=1
            try:
                token = tokenPOS.split('_')[0]
                POS = tokenPOS.split('_')[1]
                try:
                    if naivePOSdict[token] != POS:
                        wordError+=1
                        curSentError=1
                    else:
                        pass
                except KeyError:
                    # this means that word didn't appear in training
                    if POS != defaultTag:
                        wordError+=1
                        curSentError=1
                    else:
                        pass
            except:
                # this means there's something wrong in the corpus
                pass
        sentenceError+=curSentError

        
    print(len(trainLines))
    print(len(testLines))
    print(1-sentenceError/sentenceTotal)
    print(1-wordError/wordTotal)


