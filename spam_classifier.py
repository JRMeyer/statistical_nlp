import sys
import numpy as np
import os
import re
import nltk
import argparse
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, \
     f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


def get_path_label_list(dataDir,labelsPath):
    ''' 
    Input: (1) ratings file, (2) folder containing texts
    '''
    fileLabelList=[]
    with open(labelsPath) as f:
        lines = f.readlines()
    for line in lines:
        label,path = line.split(' ')
        fullPath = os.path.join(dataDir,os.path.basename(path)).rstrip()
        fileLabelList.append((fullPath,label))
    return fileLabelList
        

def create_bag_of_words(filePaths, bigrams=True, pos=True):
    rawBagOfWords = []
    regex = re.compile("X-Spam.*\n")
    for filePath in filePaths:
        with open(filePath, encoding ="latin-1") as f:
            raw = f.read()
            raw,num = re.subn(regex,'',raw)
            tokens = nltk.word_tokenize(raw)
            for token in tokens:
                rawBagOfWords.append(token)

            if pos==True:
                pos_tokens = nltk.pos_tag(raw)
                for pos_token in pos_tokens:
                    pos_token = pos_token[0]+" "+pos_token[1]
                    rawBagOfWords.append(pos_token)

            if bigrams==True:
                _bigrams = nltk.bigrams(tokens)
                for bigram in _bigrams:
                    bigram = bigram[0]+" "+bigram[1]
                    rawBagOfWords.append(bigram)

    return rawBagOfWords


def get_feature_matrix(filePaths, featureDict, bigrams=True, pos=True):
    '''
    create feature/x matrix from multiple text files
    rows = files, cols = features
    '''
    featureMatrix = np.zeros(shape=(len(filePaths),
                                      len(featureDict)),
                               dtype=float)
    regex = re.compile("X-Spam.*\n")
    for i,filePath in enumerate(filePaths):
        with open(filePath, encoding ="latin-1") as f:
            _raw = f.read()
            # in case we wanna see how many subs were made, print num
            raw,num = re.subn(regex,'',_raw)
            tokens = nltk.word_tokenize(raw)
            fileUniDist = Counter(tokens)
            for key,value in fileUniDist.items():
                if key in featureDict:
                    featureMatrix[i,featureDict[key]] = value

            if pos==True:
                pos_tokens = [(pos_token[0]+" "+pos_token[1])
                              for pos_token in nltk.pos_tag(raw)]
                filePOSdict = Counter(pos_tokens)
                for key,value in filePOSdict.items():
                    if key in featureDict:
                        featureMatrix[i,featureDict[key]] = value

            if bigrams==True:
                _bigrams = [(bigram[0]+" "+bigram[1])
                            for bigram in nltk.bigrams(tokens)]
                fileBiDist = Counter(_bigrams)
                for key,value in fileBiDist.items():
                    if key in featureDict:
                        featureMatrix[i,featureDict[key]] = value
            
    return featureMatrix

def regularize_vectors(featureMatrix):
    '''
    given a matrix, of docs and features, divide each feature value by the 
    total number of features
    '''
    for doc in range(trainX.shape[0]):
        totalWords = np.sum(trainX[doc,:],axis=0)
        trainX[doc,:] = np.multiply(trainX[doc,:],(1/totalWords))
    return featureMatrix


def get_cos_similarities(featureMatrix1,featureMatrix2,labelMatrix,stage):
    for doc in range(featureMatrix1.shape[0]):
        maxSpam=0
        maxHam=0
        for otherDoc in range(featureMatrix2.shape[0]):
            if stage=='training':
                if doc == otherDoc:
                    pass
                else:
                    _label = labelMatrix[otherDoc]
                    _cosine = np.dot(featureMatrix1[doc,:].T,
                                     featureMatrix2[otherDoc,:])
                    if _label == 1:
                        if _cosine > maxSpam:
                            maxSpam = _cosine
                    elif _label == 0:
                        if _cosine > maxHam:
                            maxHam = _cosine
            else:
                _label = labelMatrix[otherDoc]
                _cosine = np.dot(featureMatrix1[doc,:].T,
                                 featureMatrix2[otherDoc,:])
                if _label == 1:
                    if _cosine > maxSpam:
                        maxSpam = _cosine
                elif _label == 0:
                    if _cosine > maxHam:
                        maxHam = _cosine                  
        docCosines = np.ndarray(shape=(1,2),
                                buffer = np.array([maxHam,maxSpam]),
                             dtype=float)
        if doc==0:
            allCosines = docCosines
        else:
            allCosines = np.concatenate((allCosines,docCosines),axis=0)
    return allCosines


def evaluate_classifier(trueY, predY, labels, pos_label):
    print("precision score = ", str(precision_score(trueY,predY,labels=labels,
                                                    pos_label=pos_label)))
    print("accuracy score = " , str(accuracy_score(trueY,predY)))
    print("recall score = "   , str(recall_score(trueY,predY,labels=labels,
                                                 pos_label=pos_label)))
    print("f1 score = "       , str(f1_score(trueY,predY,labels=labels,
                                             pos_label=pos_label)))
    print(classification_report(trueY, predY, target_names=[str(label) for label
                                                            in labels]))

def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data', type=str, help='the directory of emails')
    parser.add_argument('-l','--labels', type=str,
                        help='the file of labels')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # get user input
    args = parse_user_args()
    dataDir = args.data
    labelsPath= args.labels

    # load in all data and labels
    pathLabelList = get_path_label_list(dataDir,labelsPath)
    allX,allY = zip(*pathLabelList)
    allY=list(allY)

    # convert labels into 0/1 for ease-of-use later
    for i,label in enumerate(allY):
        label = label.lower()
        if label=='spam':
            allY[i]=1
        if label=='ham':
            allY[i]=0
    
    # split into train and dev
    n = 1000
    k = 15
    trainPaths = allX[0:n]
    trainY = np.asarray(allY[0:n])
    devPaths = allX[n:2*n]
    devY = np.asarray(allY[n:2*n])
    testPaths = allX[2*n:3*n]
    testY = np.asarray(allY[2*n:3*n])
    
    # create feature dictionary of n-grams
    bagOfWords = create_bag_of_words(trainPaths)
    freqDist = Counter(bagOfWords)
    newBagOfWords=[]
    # throw out low freq words
    for word,freq in freqDist.items():
        if freq > k:
            newBagOfWords.append(word)
    features = set(newBagOfWords)
    featureDict = {feature:i for i,feature in enumerate(features)}

    # make feature matrices & regularize length
    trainX = get_feature_matrix(trainPaths,featureDict)
    trainX = regularize_vectors(trainX)
    devX = get_feature_matrix(devPaths,featureDict)
    devX = regularize_vectors(devX)
    testX = get_feature_matrix(testPaths,featureDict)
    testX = regularize_vectors(testX)

    # get and append max cosine similarities for spam and ham for each vector
    trainCosines = get_cos_similarities(trainX,trainX,trainY,stage='training')
    devCosines = get_cos_similarities(devX,trainX,trainY,stage='dev')
    testCosines = get_cos_similarities(testX,trainX,trainY,stage='test')
    
    finalTrainX = np.concatenate((trainX,trainCosines),axis=1)
    finalDevX = np.concatenate((devX,devCosines),axis=1)
    finalTestX = np.concatenate((testX,testCosines),axis=1)
    
    # make sparse matrices
    sparseTrainX = coo_matrix(finalTrainX)
    sparseDevX = coo_matrix(finalDevX)
    sparseTestX = coo_matrix(finalTestX)

    
    trueY = testY
    toPredictX = sparseTestX
    
    print('\n## MULTI BAYES ##')
    MNB = MultinomialNB()
    MNB.fit(sparseTrainX,trainY)
    MNBY = MNB.predict(toPredictX)
    evaluate_classifier(trueY, MNBY, labels=[1,0], pos_label=1)

    print('\n## GAUSSIAN BAYES ##')
    GNB = GaussianNB()
    GNB.fit(sparseTrainX.toarray(),trainY)
    GNBY = GNB.predict(toPredictX.toarray())
    evaluate_classifier(trueY, GNBY, labels=[1,0], pos_label=1)

    print('\n## BERNOULLI BAYES ##')
    BNB = BernoulliNB()
    BNB.fit(sparseTrainX.toarray(),trainY)
    BNBY = BNB.predict(toPredictX.toarray())
    evaluate_classifier(trueY, BNBY, labels=[1,0], pos_label=1)

    print('\n## LINEAR SVC ##')
    linSVC = LinearSVC()
    linSVC.fit(sparseTrainX,trainY)
    SVCY = linSVC.predict(toPredictX)
    evaluate_classifier(trueY, SVCY, labels=[1,0], pos_label=1)

    
    print('\n## LOGISTIC REGRESSION ##')
    logReg = LogisticRegression()
    logReg.fit(sparseTrainX,trainY)
    LRY = logReg.predict(toPredictX)
    evaluate_classifier(trueY, LRY, labels=[1,0], pos_label=1)

    print('\n## ENSEMBLE ##')
    ensemble = np.concatenate((SVCY.reshape(n,1),
                               MNBY.reshape(n,1),
                               GNBY.reshape(n,1),
                               BNBY.reshape(n,1),
                               LRY.reshape(n,1)),
                              axis=1)

    ensemble = np.sum(ensemble,axis=1)
    ensemble[ensemble <= 2] = 0
    ensemble[ensemble >= 3] = 1
    evaluate_classifier(trueY, ensemble, labels=[1,0], pos_label=1)
