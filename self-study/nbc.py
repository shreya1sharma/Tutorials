# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:15:54 2017

@author: hoge
"""

import csv
def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i]=[float(x) for x in dataset[i]]
    return dataset
    
filename = 'pima-indians-diabetes.csv'
dataset = loadCsv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(dataset) )