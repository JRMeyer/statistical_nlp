'''
Joshua Meyer

This script outputs frequency info from some Corpus
in the form of a text file. 

INPUT: path to corpus text file
OUTPUT: output.txt

USAGE: $ python3 q1a.py path/to/corpus.txt
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
        words.append(word)

# create file object for output
outfile = open('output.txt', 'w')

for item in Counter(words).most_common():
    outfile.write(item[0] +', ' +str(item[1]) + '\n')
