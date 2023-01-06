#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Oct  9 17:09:33 2022

@class: CS540
@author: Changjae Han
@Assignment: HW4
@File: hw4.py
"""

import csv
import numpy as np
from numpy import linalg as LA
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt



"""
load_data
loading data from file: filepath or file
"""
def load_data(filepath):
    
    dic_list = []
    with open('Pokemon.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #remove other properties
            row.pop('#')
            row.pop('Name')
            row.pop('Type 1')
            row.pop('Type 2')
            row.pop('Total')
            row.pop('Generation')
            row.pop('Legendary')
            dic_list.append(row)
            
    return dic_list

"""
calc_features
calculate features with 6 different traits:
    1. x_1 = Attack
    2. x_2 = Sp. Atk
    3. x_3 = Speed
    4. x_4 = Defense
    5. x_5 = Sp. Def
    6. x_6 = HP
"""
def calc_features(row):
    
    tempArray = []  
    tempArray.append(row['Attack'])
    tempArray.append(row['Sp. Atk'])
    tempArray.append(row['Speed'])
    tempArray.append(row['Defense'])
    tempArray.append(row['Sp. Def'])
    tempArray.append(row['HP'])

    #create numpy array dtype int64
    numArray = np.array(tempArray, dtype=('int64'))
    
    return numArray

"""
hac
it mimics behavior of linkage() from Scipy to show dendrogram
"""
def hac(features):
    
    n = len(features)
    
    #create array that mimics linkage(), n-1*4
    zArray = []
    for q in range(n-1):
        zArray.append([])
        for w in range(4):
            zArray[q].append(0)
    
    #normStore having every clusters by assigning it as an individual list
    normStore = []   
    for i in range(n):
        cluster = []
        cluster.append(features[i])
        normStore.append(cluster)
    
    #loop: check minimum distance while iterating through normStore row by row
    #      and after finding it, save index and add to normStore
    k = 0
    while k < n-1:

        mini = 1000000.0
        firstIn = 0
        secondIn = 0
        for j in range(len(normStore)):
            if (normStore[j] != 0):
                for t in range(j+1,len(normStore)):
                     if (normStore[t] != 0):
                         
                         #complete-linkage algorithm
                         tempDis = 0.0
                         for x in range(len(normStore[j])):
                                 for y in range(len(normStore[t])):
                                         d = LA.norm(normStore[j][x]-normStore[t][y])
                                         if (d > tempDis):
                                             tempDis = d
                         #tie breaking
                         if (tempDis < mini):
                            mini = tempDis
                            size = len(normStore[j])+len(normStore[t])
                            if(j<t):
                                firstIn = j
                                secondIn = t
                            else:
                                firstIn = t
                                secondIn = j
        
        zArray[k][0] = firstIn
        zArray[k][1] = secondIn
        zArray[k][2] = mini
        zArray[k][3] = size

        #save normstore value since it can be deleted
        A = normStore[firstIn]
        B = normStore[secondIn]
        
        del normStore[firstIn]
        normStore.insert(firstIn,0)
        del normStore[secondIn]
        normStore.insert(secondIn,0)
        normStore.append(A+B)
        
        k = k+1
        
    return np.array(zArray)

"""
imshow_hac
show dendrogram for numpy array
"""
def imshow_hac(Z):
    plt.figure()
    dendrogram(Z)
    plt.show()


