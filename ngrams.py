'''
Joshua Meyer

Given a corpus (text file), output a model of n-grams

USAGE: $ python3 ngrams.py
'''

from tokenize_corpus import *
import operator
from collections import Counter
import numpy as np
import re




def get_user_input():
    fileName = input("Enter filepath here: ")
    smoothing = input("Pick your flavor of smoothing...\n"+
                      "Enter 'laplace' or 'lidstone' or 'none' here: ")
    if smoothing == 'lidstone':
        _lambda = float(input("Choose your value for Lidstone's Lambda\n"+
                      "Enter a decimal here: "))
    else:
        _lambda = None
    return fileName, smoothing, _lambda


def tokenize_line(line):
    '''
    (1) lower all text, strip whitespace and newlines
    (2) replace everything that isn't a letter or space
    (3) split line on whitespace
    (4) pad the line
    (5) return tokens
    '''
    line = line.lower().strip().rstrip()
    # regex pattern to match everything that isn't a letter
    pattern = re.compile('[\W_0-9]+', re.UNICODE)
    # replace everything that isn't a letter or space
    line = (' ').join([pattern.sub('', token) for token in line.split(' ')])
    tokens=[]
    for token in line.split(' '):
        if token == '':
            pass
        else:
            tokens.append(token)   
    # pad the line
    tokens = ['<s>'] + tokens + ['</s>']      
    return tokens


def get_lists_of_tokens_from_file(fileName,kyrgyzLetters):
    with open(fileName) as inFile:
        content = inFile.read()
        
        unigrams=[]
        bigrams=[]
        numSentences=0
        for line in content.split('.'):
            tokens=[]
            line = tokenize_line(line)
            if line == ['<s>','</s>']:
                pass
            else:
                for token in line:
                    if (all(char in kyrgyzLetters for char in token)):
                        tokens.append(token)
                    elif token == '<s>' or token == '</s>':
                        tokens.append(token)
                    else:
                        pass
                
            for unigram in get_ngrams(tokens,1):
                unigrams.append(unigram)
                
            for bigram in get_ngrams(tokens,2):
                bigrams.append(bigram)
            numSentences+=1
    return unigrams, bigrams, numSentences


def get_ngrams(tokens, n):
    '''
    Given a list of tokens, return a list of tuple ngrams
    '''
    ngrams=[]
    # special case for unigrams
    if n==1:
        for token in tokens:
            # we need parentheses and a comma to make a tuple
            ngrams.append((token,))
    # general n-gram case
    else:
        for i in range(len(tokens)-(n-1)):
            ngrams.append(tuple(tokens[i:i+n]))
    return ngrams


def get_prob_dict(ngrams, ngramOrder, smoothing, _lambda):
    '''
    Make a dictionary of probabilities, where the key is the ngram.
    Without smoothing, we have: p(X) = freq(X)/NUMBER_OF_NGRAMS
    '''
    if smoothing == 'none':
        numSmooth = 0
        denominator =  len(ngrams)
        probUnSeen = 1/denominator
        
    elif smoothing == "laplace":
        numSmooth = 1
        denominator =  (len(ngrams)+ len(ngrams)**ngramOrder)
        probUnSeen = 1/denominator
        
    elif smoothing == 'lidstone':
        numSmooth = _lambda
        denominator = (len(ngrams) + (len(ngrams)**ngramOrder)*_lambda)
        probUnSeen = _lambda/denominator

    probDict={}
    for key, value in Counter(ngrams).items():
        probDict[key] = ((value + numSmooth) / denominator)

    return probDict, probUnSeen


def get_ngram_model(probDict):
    '''
    Given a dictionary of nGrams for some corpus, compute the conditional
    probabilities for higher order nGrams. There must be all nGrams already in
    the dict leading up to the highest order. IE, if the dict has trigrams, it
    *must* also have bigrams and unigrams. 

    p(N|N_MINUS_ONE) = log(p(N)/p(N_MINUS_ONE))

    p(A) = log(p(A))
    p(B|A) =  log(p(A_B)/p(A))
    p(C|A_B) =  log(p(A_B_C)/p(A_B))
    '''
    loggedProbs={}
    for nGram, nGramProb in probDict.items():
        if len(nGram) == 1:
            loggedProbs[nGram] = np.log10(nGramProb)
        else:
            nMinus1Gram = nGram[:(len(nGram)-1)]
            nMinus1GramProb = probDict[nMinus1Gram]
            
            condNgramProb = np.log(nGramProb) - np.log10(nMinus1GramProb)
            loggedProbs[nGram] = condNgramProb

    return loggedProbs



def print_joint_prob(fileName, probDict, probUnSeen):
    '''
    make predictions on test corpus, given probabilities from training corpus
    '''
    f = open(fileName)
    for line in f:
        probs=[]
        tokens = tokenize_line(line,2,tags=False)
        for bigram in get_ngrams(tokens,2):
            try:
                probs.append(probDict[bigram])
            except:
                probs.append(np.log10(probUnSeen))
        print(np.prod(probs))

        
kyrgyzLetters = ['а','о','у','ы','и','е','э',
                'ө','ү','ю','я','ё','п','б',
                'д','т','к','г','х','ш','щ',
                'ж','з','с','ц','ч','й','л',
                'м','н','ң','ф','в','р','ъ',
                'ь']

        
if __name__ == "__main__":
    fileName,smoothing,_lambda = get_user_input()

    unigrams,bigrams,numSentences = get_lists_of_tokens_from_file(fileName,
                                                                  kyrgyzLetters)

    uniProbDict, uniProbUnSeen = get_prob_dict(unigrams,1,smoothing,_lambda)
    biProbDict, biProbUnSeen = get_prob_dict(bigrams,2,smoothing,_lambda)

    # we need this nGramProbDict to get conditional probabilities
    ### This should be re-written without this intermediate step!
    nGramProbDict = {}
    for d in [uniProbDict,biProbDict]:
        for key,value in d.items():
            nGramProbDict[key] = value

    condProbDict = get_ngram_model(nGramProbDict)
    biCondDict={}
    for key,value in condProbDict.items():
        if len(key)==2:
            biCondDict[key]=value
                
    with open('output.txt', 'w', encoding = 'utf-8') as outFile:
        
        outFile.write('\n\data\\\n')
        outFile.write('ngram 1=' + str(len(uniProbDict)) +'\n')
        outFile.write('ngram 2=' + str(len(biCondDict)) +'\n\n')

        ## unigrams
        outFile.write('\\1-grams:\n')
        sortedUni = sorted(uniProbDict.items(), key=operator.itemgetter(1),
                           reverse=True)
        for key,value in sortedUni:
            entry = (str(np.log10(value)) +' '+ key[0])
            outFile.write(entry+'\n')

        ## bigrams
        outFile.write('\n\\2-grams:\n')
        sortedBi = sorted(biCondDict.items(), key=operator.itemgetter(1),
                           reverse=True)
        for key,value in sortedBi:
            entry = (str(value) +' '+ key[0] +' '+ key[1])
            outFile.write(entry+'\n')

        outFile.write('\n\end\\')
