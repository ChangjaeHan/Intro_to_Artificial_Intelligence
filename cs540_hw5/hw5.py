#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 15:47:58 2022

@author: changjae han
"""
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy.linalg import inv
  

#2
data = open(sys.argv[1])

size = 0
Years = []
Days = []

#skip the first line
next(data)

#read
for line in data:
    (year, days) = line.split(',')       
    Years.append(int(year))
    Days.append(int(days))
    size = size+1

plt.plot(Years,Days)


plt.xlabel('Year')
plt.ylabel('Number of frozen days')

plt.savefig("plot.jpg")
plt.show()
data.close()


#3.a
xi = []
for i in range(size):
    xi.append([])
    xi[i].append(1)
    xi[i].append(Years[i])

X = np.array(xi)
print("Q3a:")
print(X)

#3.b
y = []
for i in range(size):
    y.append(Days[i])
    
Y = np.array(y)
print("Q3b:")
print(Y)

    
#3.c
Z = np.dot(np.transpose(X),X)
print("Q3c:")
print(Z)
   
#3.d
I = inv(Z)
print("Q3d:")
print(I)

#3.e
PI = np.dot(I,np.transpose(X))
print("Q3e:")
print(PI)

#3.f
hat_beta = np.dot(PI,Y)
print("Q3f:")
print(hat_beta)

#4
y_test = hat_beta[0] + np.dot(hat_beta[1],2021) 
print("Q4: " + str(y_test))

#5
if hat_beta[1] > 0 :
    symbol = ">"
elif hat_beta[1] < 0 :
    symbol = "<"
elif hat_beta[1] == 0 :
    symbol = "="

print("Q5a: " + symbol)
print("Q5b: for > the number of frozen days are likely to increase, for < are likely to decrease, for = are likely to be constant ")


#6
xStar = -hat_beta[0]/hat_beta[1]
print("Q6a:" + str(xStar))
print("Q6b: it seems that x* is a quite compelling prediction since based on the trends in the data, frozen days tend to decrease per year and MLE shows reasonable value that converges to y = 0") 

