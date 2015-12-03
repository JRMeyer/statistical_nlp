'''
Joshua Meyer

This script outputs frequency info from some Corpus
in the form of a text file. 

INPUT: path to corpus text file
OUTPUT: output.txt

USAGE: $ python3 freq_info_from_corpus.py path/to/corpus.txt
'''

import sys
from collections import Counter

# open up the corpus
fileName = sys.argv[1]
f = open(fileName)

words = []

# tokenize on whitespace
for line in f:
    line = line.rstrip().lower()
    for word in line.split(' '):
        if word == '':
            pass
        else:
            words.append(word)

# create file object for output
outFile = open('output.txt', 'w')

outFile.write('A total of '+ str(len(words)) +' tokens were in the text\n')
for item in Counter(words).most_common():
    outFile.write(item[0] +', ' +str(item[1]) + '\n')
