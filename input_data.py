import glob
import random


def input_data(hamDir,spamDir,percentTest):
    ''' 
    Input:
      hamDir: String. dir of ham text files
      spamDir: String. dir of spam text file
      percentTest: Float. percentage of all data to be assigned to testset
    Returns:
      trainPaths: Array. Absolute paths to training emails
      trainY: Array. Training labels, 0 or 1 int.
      testPaths: Array. Absolute paths to testing emails
      testY: Array. Testing labels, 0 or 1 int.
    '''
    pathLabelPairs={}
    for hamPath in glob.glob(hamDir+'*'):
        pathLabelPairs.update({hamPath:0})
    for spamPath in glob.glob(spamDir+'*'):
        pathLabelPairs.update({spamPath:1})
    
    # get test set as random subsample of all data
    numTest = int(percentTest * len(pathLabelPairs))
    testing = set(random.sample(pathLabelPairs.items(),numTest))

    # delete testing data from superset of all data
    for entry in testing:
        del pathLabelPairs[entry[0]]
    
    # split training tuples of (path,label) into separate lists
    trainPaths=[]
    trainY=[]
    for item in pathLabelPairs.items():
        trainPaths.append(item[0])
        trainY.append(item[1])
    del pathLabelPairs

    # split testing tuples of (path,label) into separate lists
    testPaths=[]
    testY=[]
    for item in testing:
        testPaths.append(item[0])
        testY.append(item[1])
    del testing

    return trainPaths, trainY, testPaths, testY

def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ham','--hamDir')
    parser.add_argument('-spam','--spamDir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import sys, argparse
    # get user input
    args = parse_user_args()
    hamDir = args.hamDir
    spamDir= args.spamDir

    trainPaths,trainY,testPaths,testY = input_data(hamDir,spamDir,.1)

    print(len(testPaths), len(trainPaths))
