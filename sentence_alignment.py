'''
Assuming that each line in a file corresponds to a sentence,
take two files which already have their sentences aligned and
then align their words using word frequency as a cost metric
'''

import argparse
import numpy as np
import operator

def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f1','--infile1', type=str, required=True,
                        help='the first input text file')
    parser.add_argument('-f2','--infile2', type=str, required=True,
                        help='the second input text file')
    parser.add_argument('-u','--unit',type=str,
                        help='unit for alignment metric',
                        choices = ['word','character'],
                        default='word')
    args = parser.parse_args()
    return args

def get_lines_from_file(fileName):
    with open(fileName) as f:
        lines = f.readlines()
    return lines

def get_lengths_of_lines(lines, unit):
    lengths=[]
    for line in lines:
        if unit == 'word':
            length=len(line.split(' '))
        elif unit == 'character':
            length = len(line)
        lengths.append(length)
    return lengths

def print_latex_table_lengths(lengths1,lengths2,unit):
    print(r'\begin{table}[H]')
    print(r'\centering')
    print(r'\begin{tabular}{cc}')
    print(r'\toprule')
    print(r'Sentence    & Number of ' + unit +'s' +r'\\')
    print(r'\midrule')
    
    for i,length in enumerate(lengths1):
        print('s' + str(i+1) +'&'+ str(length) +r'\\')
    print(r'\midrule')
    for i,length in enumerate(lengths2):
        print('t' + str(i+1) +'&'+ str(length) +r'\\')
        
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')

    
def get_min_distance(i, j, lengthDict1, lengthDict2, Trellis):
    distances={}
    try:
        # 0:1
        D = Trellis[i,j-1]
        cost = lengthDict2[j]
        distance = np.sqrt((D+cost)**2)
        if distance in distances:
            distances[distance]+=['01']
        else:
            distances[distance]=['01']      
    except KeyError:
        pass

    try:
        # 1:0
        D = Trellis[i-1,j]
        cost = lengthDict1[i]
        distance = np.sqrt((D+cost)**2)
        if distance in distances:
            distances[distance]+=['10']
        else:
            distances[distance]=['10']
    except KeyError:
        pass

    try:
        # 1:1
        D = Trellis[i-1,j-1]
        cost = lengthDict1[i]-lengthDict2[j]
        distance = np.sqrt((D+cost)**2)
        if distance in distances:
            distances[distance]+=['11']
        else:
            distances[distance]=['11']
    except KeyError:
        pass

    try:
        # 1:2
        D = Trellis[i-1,j-2]
        cost = lengthDict1[i] - (lengthDict2[j]+lengthDict2[j-1])
        distance = np.sqrt((D+cost)**2)
        if distance in distances:
            distances[distance]+=['12']
        else:
            distances[distance]=['12']
    except KeyError:
        pass


    try:
        # 2:1
        D = Trellis[i-2,j-1]
        cost = (lengthDict1[i]+lengthDict1[i-1]) - lengthDict2[j]
        distance = np.sqrt((D+cost)**2)
        if distance in distances:
            distances[distance]+=['21']
        else:
            distances[distance]=['21']
    except KeyError:
        pass

    try:
        # 2:2
        D = Trellis[i-2,j-2]
        cost = ((lengthDict1[i]+lengthDict1[i-1]) -
                (lengthDict2[j]+lengthDict2[j-1]))
        distance = np.sqrt((D+cost)**2)
        if distance in distances:
            distances[distance]+=['22']
        else:
            distances[distance]=['22']
    except KeyError:
        pass

    # sort the list and take the first, and hence least, entry
    minItem = sorted(distances.items(), key=operator.itemgetter(1),
                      reverse=True)[0]
    minValue = minItem[0]
    minPaths = minItem[1]
    return minValue, minPaths


def get_length_dict(lengths):
    lengthDict={}
    for i,length in enumerate(lengths):
        i+=1
        lengthDict[i]=length
    return lengthDict

        
if __name__ == "__main__":
    # get user input
    args = parse_user_args()
    fileName1 = args.infile1
    fileName2 = args.infile2
    unit = args.unit

    lines1 = get_lines_from_file(fileName1)
    lines2 = get_lines_from_file(fileName2)

    lengths1 = get_lengths_of_lines(lines1,unit)
    lengths2 = get_lengths_of_lines(lines2,unit)

    lengthDict1 = get_length_dict(lengths1)
    lengthDict2 = get_length_dict(lengths2)

    # add first state zero
    lengthDict1[0]=0
    lengthDict2[0]=0
    
    # create matrix to hold local probailities
    Trellis = np.zeros([len(lengthDict1),
                        len(lengthDict2)],dtype=float)

    pathDict={}
    for i in lengthDict1:
        for j in lengthDict2:
            minValue, minPaths = get_min_distance(i,j,lengthDict1,
                                                  lengthDict2,Trellis)
            Trellis[i,j] = minValue
            pathDict[(i,j)] = minPaths

    print(Trellis)
    numCols = Trellis.shape[1]
    
    nextRow = Trellis.argmin(axis=0)[-1]
    print(nextRow)
    nextCol = numCols-1
    while nextCol > 0:
        path = (pathDict[(nextRow,nextCol)][0])
        print(path)
        nextRow = nextRow+int(path[0])
        nextCol = nextCol-int(path[1])
