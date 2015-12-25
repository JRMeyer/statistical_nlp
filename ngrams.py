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


def get_nGram_tuples(lines,startTime,lenSentenceCutoff):
    unigrams=[]
    bigrams=[]
    trigrams=[]
    for line in lines.split('\n'):
        line = line.split(' ')
        if len(line) >lenSentenceCutoff:
            unigrams+=get_nGrams_from_line(line,1)
            bigrams+=get_nGrams_from_line(line,2)
            trigrams+=get_nGrams_from_line(line,3)
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' A total of ' +str(len(unigrams))+
          ' unigrams found')
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+ ' A total of '+
          str(len(bigrams)) + ' bigrams found')
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+ ' A total of '+
          str(len(trigrams)) + ' trigrams found')
    return unigrams, bigrams, trigrams


def get_nGrams_from_line(tokens, n):
    '''
    Given a list of tokens, return a list of tuple nGrams
    '''
    nGrams=[]
    # special case for unigrams
    if n==1:
        for token in tokens:
            # we need parentheses and a comma to make a tuple
            nGrams.append((token,))
    # general n-gram case
    else:
        for i in range(len(tokens)-(n-1)):
            nGrams.append(tuple(tokens[i:i+n]))
    return nGrams


def get_freq_dict(nGrams, nGramOrder, smoothing, _lambda, startTime):
    '''
    Make a dictionary of probabilities, where the key is the nGram.
    Without smoothing, we have: p(X) = freq(X)/NUMBER_OF_NGRAMS
    '''
    
    if smoothing == 'none':
        denominator = len(nGrams)
        probDict={}
        for key, value in Counter(nGrams).items():
            probDict[key] = value/denominator

    elif smoothing == "laplace":
        numSmooth = 1
        denominator = (len(nGrams)+ len(nGrams)**nGramOrder)
        probUnSeen = 1/denominator
        probDict={}
        for key, value in Counter(nGrams).items():
            probDict[key] = (value + numSmooth) / denominator

    elif smoothing == 'lidstone':
        numSmooth = _lambda
        denominator = (len(nGrams) + (len(nGrams)**nGramOrder)*_lambda)
        probUnSeen = _lambda/denominator
        probDict={}
        for key, value in Counter(nGrams).items():
            probDict[key] = (value + numSmooth) / denominator

    elif smoothing == 'turing':
        # N = total number of n-grams in text
        # N_1 = number of hapaxes
        # r = frequency of an n-gram
        # n_r = number of n-grams which occured r times
        # P_T = r*/N, the probability of an n-gram which occured r times
        # r* = (r+1) * ( (n_{r+1}) / (n_r) )

        N = len(unigrams)
        
        freqDist_nGrams={}
        for key,value in Counter(nGrams).items():
            # key = n-gram, value = r
            freqDist_nGrams[key] = value
            
        freqDist_freqs={}
        for key,value in Counter(freqDist_nGrams.values()).items():
            # key = r, value = n_r
            freqDist_freqs[key] = value
        
        probDict={}
        for key,value in freqDist_nGrams.items():
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
            
        
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t] '+
          str(nGramOrder) + '-gram probability dictionary made')
    
    return probDict


def get_conditional_dict(nMinus1GramDict,nGramDict,nGramOrder):
    '''
    Given a dictionary of nGrams and one of n-1-grams with their 
    frequencies for some corpus, compute the conditional probabilities for
    nGrams.
    
    p(N|N_MINUS_ONE) = p(N)/p(N_MINUS_ONE)
    p(B|A) =  p(A_B)/p(A)
    log(p(B|A)) = log(p(A_B)) - log(p(A))
    '''
    nGramCondDict={}
    for nGram, nGramProb in nGramDict.items():
        nMinus1Gram = nGram[:nGramOrder-1]
        nMinus1GramProb = nMinus1GramDict[nMinus1Gram]
        condNGramProb = nGramProb / nMinus1GramProb
        nGramCondDict[nGram] = condNGramProb
    return nGramCondDict


def get_bow_katz_dict(uniProbDict,biProbDict,triProbDict,startTime):
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
    
    for biKey,biValue in biProbDict.items():
        numerator=0
        denominator=0
        for triKey,triValue in triProbDict.items():
            if triKey[0:2] == biKey[:]:
                numerator += triValue
                denominator += biValue
        alpha = 1-(numerator/denominator)
        bowDict[biKey] = alpha*biValue
        
    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' 3-gram backoff model made')
    return bowDict


def get_brants_bow_dict(nGramDict, alpha=.4):
    bow_dict={}
    for key,value in nGramDict.items():
        bow_dict[key] = (value,alpha*value)
    return bow_dict
    
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

    # make the cutOff
    cutOffWords = get_cutOff_words(tokens,k,startTime)
    lines = replace_cutoff_with_UNK(lines,cutOffWords,startTime)


    ## START HERE if you want to feed in a cleaned file with <UNK>
    # with open(fileName) as f:
    #     lines = f.read()
    
    # get lists of tuples of nGrams
    unigrams, bigrams, trigrams = get_nGram_tuples(lines,startTime,
                                                   lenSentenceCutoff=4)

    # get probability dictionaries
    uniProbDict = get_freq_dict(unigrams,1,smoothing,_lambda,startTime)
    biProbDict = get_freq_dict(bigrams,2,smoothing,_lambda,startTime)
    triProbDict = get_freq_dict(trigrams,3,smoothing,_lambda,startTime)

    biCondDict = get_conditional_dict(uniProbDict,biProbDict,2)
    triCondDict = get_conditional_dict(biProbDict,triProbDict,3)

    uniBowDict = get_brants_bow_dict(uniProbDict)
    biBowDict = get_brants_bow_dict(biCondDict)
    
    if backoff:
        backedOff = 'yes'
    else:
        backedOff = 'no'
        
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
        sortedUni = sorted(uniBowDict.items(), key=operator.itemgetter(1),
                          reverse=True)
        
        for key,value in sortedUni:
            if backoff:
                entry = (str(np.log(value[0])) +' '+ key[0] +' '+
                         str(np.log(value[1])))
            else:
                entry = (str(np.log(value[0])) +' '+ key[0])
            outFile.write(entry+'\n')

            
        ## print bigrams
        outFile.write('\n\\2-grams:\n')
        sortedBi = sorted(biBowDict.items(), key=operator.itemgetter(1),
                           reverse=True)

        for key,value in sortedBi:
            if backoff:
                entry = (str(np.log(value[0])) +' '+ key[0] +' '+ key[1] +' '+
                         str(np.log(value[1])))
            else:
                entry = (str(np.log(value[0])) +' '+ key[0] +' '+ key[1])
            outFile.write(entry+'\n')


        ## print trigrams
        outFile.write('\n\\3-grams:\n')
        sortedTri = sorted(triCondDict.items(), key=operator.itemgetter(1),
                           reverse=True)
        
        for key,value in sortedTri:
            entry = (str(np.log(value)) +' '+ key[0] +' '+ key[1] +' '+ key[2])
            outFile.write(entry+'\n')
            
        outFile.write('\n\end\\')

    print('[  '+ str("%.2f" % (time.time()-startTime)) +'  \t]'+
          ' successfully printed model to file!')
