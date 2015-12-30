import sys
import numpy as np
from input_data import input_data
import os
import re
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

        

def create_bag_of_words(filePaths):
    '''
    Input:
      filePaths: Array. A list of absolute filepaths
    Returns:
      bagOfWords: Array. All tokens in files
    '''
    bagOfWords = []
    regex = re.compile("X-Spam.*\n")
    for filePath in filePaths:
        with open(filePath, encoding ="latin-1") as f:
            raw = f.read()
            raw = re.sub(regex,'',raw)
            tokens = raw.split()
            for token in tokens:
                bagOfWords.append(token)
    return bagOfWords


def get_feature_matrix(filePaths, featureDict):
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
            raw = re.sub(regex,'',_raw)
            tokens = raw.split()
            fileUniDist = Counter(tokens)
            for key,value in fileUniDist.items():
                if key in featureDict:
                    featureMatrix[i,featureDict[key]] = value
    return featureMatrix

def regularize_vectors(featureMatrix):
    '''
    Input:
      featureMatrix: matrix, where docs are rows and features are columns
    Returns:
      featureMatrix: matrix, updated by dividing each feature value by the total
      number of features for a given document
    '''
    for doc in range(featureMatrix.shape[0]):
        totalWords = np.sum(featureMatrix[doc,:],axis=0)
        featureMatrix[doc,:] = np.multiply(featureMatrix[doc,:],(1/totalWords))
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
    parser.add_argument('-ham','--hamDir', help='path to dir of ham txt files')
    parser.add_argument('-spam','--spamDir', help='dir path of spam txt files')
    args = parser.parse_args()
    return args



def demo(hamDir,spamDir,k):
    trainPaths,trainY,testPaths,testY = input_data(hamDir=hamDir,
                                                   spamDir=spamDir,
                                                   percentTest=.1)
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
    testX = get_feature_matrix(testPaths,featureDict)
    testX = regularize_vectors(testX)

    # get and append max cosine similarities for spam and ham for each vector
    trainCosines = get_cos_similarities(trainX,trainX,trainY,stage='training')
    testCosines = get_cos_similarities(testX,trainX,trainY,stage='test')
    
    finalTrainX = np.concatenate((trainX,trainCosines),axis=1)
    finalTestX = np.concatenate((testX,testCosines),axis=1)
    
    # make sparse matrices
    sparseTrainX = coo_matrix(finalTrainX)
    sparseTestX = coo_matrix(finalTestX)

    
    trueY = testY
    toPredictX = sparseTestX
    n = toPredictX.shape[0]

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


if __name__ == '__main__':
    # get user input
    args = parse_user_args()
    hamDir = args.hamDir
    spamDir = args.spamDir

    # frequency cutoff
    k=15

    demo(hamDir,spamDir,k)
