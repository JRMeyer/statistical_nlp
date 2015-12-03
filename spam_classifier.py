import numpy as np
import os
import re
import nltk
import argparse
from collections import Counter

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, \
     f1_score


def get_path_label_list(dataDir,labelsPath):
    ''' Input: (1) ratings file, (2) folder containing texts
    '''
    fileLabelList=[]
    with open(labelsPath) as f:
        lines = f.readlines()
    for line in lines:
        label,path = line.split(' ')
        fullPath = os.path.join(dataDir,os.path.basename(path)).rstrip()
        fileLabelList.append((fullPath,label))
    return fileLabelList
        

def create_bag_of_words(filePaths, bigrams=True):
    rawBagOfWords = []
    for filePath in filePaths:
        raw = open(filePath, encoding ="latin-1").read()
        tokens = nltk.word_tokenize(raw)
        for token in tokens:
            rawBagOfWords.append(token)
        if bigrams==True:
            _bigrams = nltk.bigrams(tokens)
            for bigram in _bigrams:
                bigram = bigram[0]+" "+bigram[1]
                rawBagOfWords.append(bigram)
        else:
            pass
    return rawBagOfWords



def get_feature_matrix(filePaths, featureDict, bigrams=True):
    '''
    create feature/x matrix for classifiers
    '''
    featureMatrix = np.ndarray(shape=(len(filePaths),
                                      len(featureDict)),
                               dtype=float)
    
    for i,filePath in enumerate(filePaths):
        raw = open(filePath,encoding="latin-1").read()
        tokens = nltk.word_tokenize(raw)
        fileUniDist = Counter(tokens)
        for key,value in fileUniDist.items():
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


def evaluate_classifier(trueY, predY):
    print("precision score = ", str(precision_score(trueY,predY)))
    print("accuracy score = " , str(accuracy_score(trueY,predY)))
    print("recall score = "   , str(recall_score(trueY,predY)))
    print("f1 score = "       , str(f1_score(trueY,predY)))


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
    
    pathLabelList = get_path_label_list(dataDir,labelsPath)
    allX,allY = zip(*pathLabelList)
    
    trainPaths = allX[0:1000]
    trainY = np.asarray(allY[0:1000])

    bagOfWords = create_bag_of_words(trainPaths)
    features = set(bagOfWords)
    featureDict = {feature:i for i,feature in enumerate(features)}
    
    trainX = get_feature_matrix(trainPaths,featureDict)

    # NAIVE BAYES
    multiNB = MultinomialNB()
    multiNB.fit(trainX,trainY)
    predY = multiNB.predict(trainX[:100])
    trueY = trainY[:100]
    evaluate_classifier(trueY,predY)

    # LINEAR SVC
    linSVC = LinearSVC()
    linSVC.fit(trainX,trainY)
    predY = linSVC.predict(trainX[:100])
    trueY = trainY[:100]
    evaluate_classifier(trueY,predY)
