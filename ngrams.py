'''
Joshua Meyer

Given a corpus (text file), output a model of n-grams

USAGE: $ python3 ngrams.py
'''

import argparse
import operator
from collections import Counter
import numpy as np
import re
import time
import sys

def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile', type=str, help='the input text file')
    parser.add_argument('-s','--smoothing',type=str, help='flavor of smoothing',
                        choices = ['none','laplace','lidstone','turing'],
                        default='none')
    parser.add_argument('-w','--weight', type=int, help='Lidstones lambda',
                        default=None)
    parser.add_argument('-bo','--backoff', action='store_true',
                        help='add backoff weights')
    parser.add_argument('-k','--cutoff', type=int, default=1,
                        help='frequency count cutoff')
    args = parser.parse_args()
    return args

def get_lines_from_file(fileName,kyrgyzLetters,startTime):
    regex = re.compile(r'[.!?\n]')
    with open(fileName) as inFile:
        lines=''
        content = inFile.read()
        for line in re.split(regex,content):
            line = tokenize_line(line,kyrgyzLetters)
            lines += line
    inFile.close()
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' File read and tokenized')
    return lines


def tokenize_line(line,kyrgyzLetters):
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
        # make sure we only have Kyrgyz letters in token
        elif all(char in kyrgyzLetters for char in token):
            tokens.append(token)
    # check if there are tokens, then pad the line
    if tokens:
        line = (' ').join(tokens)
        line = '<s> ' + line + ' </s>\n'
    else:
        line = ''
    return line


def get_cutOff_words(tokens,k,startTime):
    uniFreqDict = Counter(tokens)
    cutOffWords=[]
    for key,value in uniFreqDict.items():
        if value <= k:
            cutOffWords.append(key)
            
    numCutOffWords = len(cutOffWords)
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' A total of '+ str(numCutOffWords) + ' words occurring less than '+
          str(k)+ ' time(s) identified')
    return cutOffWords


def replace_cutoff_with_UNK(lines, cutOffWords, startTime):
    # string.replace() is twice as fast as re.sub()!
    for key in cutOffWords:
        lines = lines.replace(' '+key+' ',' <UNK> ')
            
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' Cutoff Words replaced with <UNK> ')
    return lines



def get_ngram_tuples(lines,startTime,lenSentenceCutoff):
    unigrams=[]
    bigrams=[]
    trigrams=[]
    for line in lines.split('\n'):
        line = line.split(' ')
        if len(line) >lenSentenceCutoff:
            unigrams+=get_ngrams_from_line(line,1)
            bigrams+=get_ngrams_from_line(line,2)
            trigrams+=get_ngrams_from_line(line,3)
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' A total of ' +str(len(unigrams))+
          ' unigrams found')
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+ ' A total of '+
          str(len(bigrams)) + ' bigrams found')
    return unigrams, bigrams, trigrams


def get_ngrams_from_line(tokens, n):
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


def get_prob_dict(ngrams, ngramOrder, smoothing, _lambda, startTime):
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

        N = len(unigrams)
        
        freqDist_ngrams={}
        for key,value in Counter(ngrams).items():
            # key = n-gram, value = r
            freqDist_ngrams[key] = value
            
        freqDist_freqs={}
        for key,value in Counter(freqDist_ngrams.values()).items():
            # key = r, value = n_r
            freqDist_freqs[key] = value
        
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

        # need to figure out this one as N/N_1...
        probUnSeen=.0001
        
    else:
        probDict={}
        for key, value in Counter(ngrams).items():
            probDict[key] = ((value + numSmooth) / denominator)

    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t] '+
          str(ngramOrder) + '-gram probability dictionary made')
    
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


def get_bow_dict(uniProbDict,biProbDict,triProbDict,startTime):
    # calculate backoff weights as in Katz 1987
    bowDict={}
    for uniKey,uniValue in uniProbDict.items():
        numerator = 0
        denominator = 0
        for biKey,biValue in biProbDict.items():
            if biKey[0] == uniKey[0]:
                numerator += biValue
                denominator += uniValue
        alpha = (1-numerator/1-denominator)
        bowDict[uniKey] = alpha*uniValue
        
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' 2-gram backoff model made')
    
    # for biKey,biValue in biProbDict.items():
    #     numerator=0
    #     denominator=0
    #     for triKey,triValue in triProbDict.items():
    #         if triKey[0:2] == biKey[:]:
    #             numerator += triValue
    #             denominator += biValue
    #     alpha = 1-(numerator/denominator)
    #     bowDict[biKey] = alpha*biValue
        
    # print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
    #       ' 3-gram backoff model made')
    return bowDict

        
kyrgyzLetters = ['а','о','у','ы','и','е','э',
                'ө','ү','ю','я','ё','п','б',
                'д','т','к','г','х','ш','щ',
                'ж','з','с','ц','ч','й','л',
                'м','н','ң','ф','в','р','ъ',
                'ь']


if __name__ == "__main__":
    # get user input
    args = parse_user_args()
    fileName = args.infile
    smoothing = args.smoothing
    _lambda = args.weight
    backoff = args.backoff
    k = args.cutoff
    
    startTime = time.time()
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+ ' running')

    # tokenize file
    lines = get_lines_from_file(fileName,kyrgyzLetters,startTime)
    tokens = [token for line in lines.split('\n') for token in line.split(' ')]

    # with open('clean_lines.txt', 'w', encoding = 'utf-8') as outlines:
    #     outlines.write(lines)
        
    # make the cutOff
    cutOffWords = get_cutOff_words(tokens,k,startTime)
    lines = replace_cutoff_with_UNK(lines,cutOffWords,startTime)


    ## START HERE if you want to feed in a cleaned file with <UNK>
    # with open(fileName) as f:
    #     lines = f.read()
    
    # get lists of tuples of ngrams
    unigrams, bigrams, trigrams = get_ngram_tuples(lines,startTime,
                                                   lenSentenceCutoff=4)

        
    # get probability dictionaries
    uniProbDict, uniProbUnSeen = get_prob_dict(unigrams,1,smoothing,_lambda,
                                               startTime)
    biProbDict, biProbUnSeen = get_prob_dict(bigrams,2,smoothing,_lambda,
                                             startTime)
    triProbDict, triProbUnSeen = get_prob_dict(trigrams,3,smoothing,_lambda,
                                             startTime)
    # get back-off weight dictionary
    if backoff:
        bowDict = get_bow_dict(uniProbDict,biProbDict,triProbDict,startTime)
        backedOff = 'true'
    else:
        bowDict = None
        backedOff = 'false'
        
    with open(('lm_smoothing-' + smoothing +
               '_backoff-' + backedOff +
               '_cutoff-' + str(k)+
               '.txt'),
              'w', encoding = 'utf-8') as outFile:

        # Print ARPA preamble
        outFile.write('\n\data\\\n')
        outFile.write('ngram 1=' + str(len(uniProbDict)) +'\n')
        outFile.write('ngram 2=' + str(len(biProbDict)) +'\n')
        outFile.write('ngram 3=' + str(len(triProbDict)) +'\n')

        ## print unigrams
        outFile.write('\n\\1-grams:\n')
        sortedUni = sorted(uniProbDict.items(), key=operator.itemgetter(1),
                          reverse=True)
        
        for key,value in sortedUni:
            if backoff:
                entry = (str(np.log(value)) +' '+ key[0] +' '+
                         str(np.log(bowDict[key])))
            else:
                entry = (str(np.log(value)) +' '+ key[0])
            outFile.write(entry+'\n')

            
        ## print bigrams
        outFile.write('\n\\2-grams:\n')
        sortedBi = sorted(biProbDict.items(), key=operator.itemgetter(1),
                           reverse=True)

        for key,value in sortedBi:
            if backoff:
                entry = (str(np.log(value)) +' '+ key[0] +' '+ key[1] +' '+
                         str(np.log(bowDict[key])))
            else:
                entry = (str(np.log(value)) +' '+ key[0] +' '+ key[1])
            outFile.write(entry+'\n')


        ## print trigrams
        outFile.write('\n\\3-grams:\n')
        sortedTri = sorted(triProbDict.items(), key=operator.itemgetter(1),
                           reverse=True)
        
        for key,value in sortedTri:
            entry = (str(np.log(value)) +' '+ key[0] +' '+ key[1] +' '+ key[2])
            outFile.write(entry+'\n')
            
        outFile.write('\n\end\\')

    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' successfully printed model to file!')
