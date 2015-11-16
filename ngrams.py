'''
Joshua Meyer

Given a corpus (text file), output a model of n-grams

USAGE: $ python3 ngrams.py
'''

import operator
from collections import Counter
import numpy as np
import re




def get_user_input():
    fileName = input("Enter filepath here: ")
    smoothing = input("Pick your flavor of smoothing...\n"+
                      "Enter 'laplace' or 'lidstone', 'turing', or 'none': ")
    if smoothing == 'lidstone':
        _lambda = float(input("Choose your value for Lidstone's Lambda\n"+
                      "Enter a decimal here: "))
    else:
        _lambda = None
    backoff = input("Generate a Backoff model?\nEnter 'true' or 'false': ")
    return fileName, smoothing, _lambda, backoff


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


def get_ngrams_from_file(fileName,kyrgyzLetters):
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


def get_prob_dict(ngrams, ngramOrder, smoothing, _lambda):
    '''
    Make a dictionary of probabilities, where the key is the ngram.
    Without smoothing, we have: p(X) = freq(X)/NUMBER_OF_NGRAMS
    '''
    if smoothing == 'none':
        numSmooth = 0
        denominator = len(ngrams)
        probUnSeen = 1/denominator
        
    elif smoothing == "laplace":
        numSmooth = 1
        denominator = (len(ngrams)+ len(ngrams)**ngramOrder)
        probUnSeen = 1/denominator
        
    elif smoothing == 'lidstone':
        numSmooth = _lambda
        denominator = (len(ngrams) + (len(ngrams)**ngramOrder)*_lambda)
        probUnSeen = _lambda/denominator

    if smoothing == 'turing':
        # N = total number of n-grams in text
        # N_1 = number of hapaxes
        # r = frequency of an n-gram
        # n_r = number of n-grams which occured r times
        # P_T = r*/N, the probability of an n-gram which occured r times
        # r* = (r+1) * ( (n_{r+1}) / (n_r) )
        
        freqDist_ngrams={}
        for key,value in Counter(ngrams).items():
            # key = n-gram, value = r
            freqDist_ngrams[key] = value
            
        freqDist_freqs={}
        for key,value in Counter(freqDist_ngrams.values()).items():
            # key = r, value = n_r
            freqDist_freqs[key] = value
            
        N = len(ngrams)
        N_1 = freqDist_freqs[1]
        probUnSeen = N_1/N
        
        probDict={}
        for key,value in freqDist_ngrams.items():
            r = value
            n_r = freqDist_freqs[r]
            try:
                n_r_plus_1 = freqDist_freqs[r+1]
            except KeyError as exception:
                print('There are no n-grams with frequency ' +str(exception)+
                      '... using ' +str(r)+ ' instead')
                n_r_plus_1 = n_r
            r_star = (r+1)*((n_r_plus_1)/(n_r))
            probDict[key] = r_star/N
        
    else:
        probDict={}
        for key, value in Counter(ngrams).items():
            probDict[key] = ((value + numSmooth) / denominator)

    return probDict, probUnSeen


def get_conditional_model(uniProbDict,biProbDict):
    '''
    Given a dictionary of unigrams and one of bigrams with their logged 
    frequencies for some corpus, compute the conditional probabilities for
    bigrams.

    p(N|N_MINUS_ONE) = p(N)/p(N_MINUS_ONE)

    p(B|A) =  p(A_B)/p(A)
    log(p(B|A)) = log(p(A_B)) - log(p(A))
    '''
    biCondDict={}
    for biGram, biGramProb in biProbDict.items():
        A=(biGram[0],)
        nMinus1GramProb = uniProbDict[A]
        if nMinus1GramProb<biGramProb:
            print(biGram,str(biGramProb),A,str(nMinus1GramProb))
        # since the freqs are already logged, subtract instead of divide
        condNgramProb = biGramProb - nMinus1GramProb
        biCondDict[biGram] = condNgramProb

    return biCondDict


def get_bow_dict(uniProbDict,biProbDict):
    # calculate backoff weights as in Katz 1987
    bowDict={}
    for uniKey,uniValue in uniProbDict.items():
        numerator=0
        denominator=0
        for biKey,biValue in biProbDict.items():
            if biKey[0] == uniKey[0]:
                numerator+=biValue
                denominator+=uniProbDict[(biKey[1],)]
        alpha = (1-numerator)/(1-denominator)
        bowDict[uniKey] = alpha*uniValue
    return bowDict

        
kyrgyzLetters = ['а','о','у','ы','и','е','э',
                'ө','ү','ю','я','ё','п','б',
                'д','т','к','г','х','ш','щ',
                'ж','з','с','ц','ч','й','л',
                'м','н','ң','ф','в','р','ъ',
                'ь']

        
if __name__ == "__main__":
    # get user input
    fileName,smoothing,_lambda,backoff = get_user_input()

    # get lists of tuples of ngrams
    unigrams,bigrams,numSentences = get_ngrams_from_file(fileName,kyrgyzLetters)
    
    # get probability dictionaries
    uniProbDict, uniProbUnSeen = get_prob_dict(unigrams,1,smoothing,_lambda)
    biProbDict, biProbUnSeen = get_prob_dict(bigrams,2,smoothing,_lambda)

    # get back-off weight dictionary
    if backoff == 'true':
        bowDict = get_bow_dict(uniProbDict,biProbDict)
    else:
        pass
    
    with open('output.txt', 'w', encoding = 'utf-8') as outFile:
        
        outFile.write('\n\data\\\n')
        outFile.write('ngram 1=' + str(len(uniProbDict)) +'\n')
        outFile.write('ngram 2=' + str(len(biProbDict)) +'\n')

        ## print unigrams
        outFile.write('\n\\1-grams:\n')
        sortedUni = sorted(uniProbDict.items(), key=operator.itemgetter(1),
                          reverse=True)
        
        for key,value in sortedUni:
            entry = (str(np.log(value)) +' '+ key[0] +' '+
                     str(np.log(bowDict[key])))
            outFile.write(entry+'\n')

        ## print bigrams
        outFile.write('\n\\2-grams:\n')
        sortedBi = sorted(biProbDict.items(), key=operator.itemgetter(1),
                           reverse=True)
        
        for key,value in sortedBi:
            entry = (str(np.log(value)) +' '+ key[0] +' '+ key[1])
            outFile.write(entry+'\n')

        outFile.write('\n\end\\')
