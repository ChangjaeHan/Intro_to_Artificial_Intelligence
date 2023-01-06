#Name: ChangjaeHan_HW2
#Professor: Yingyu Lian
import sys
import math


def get_parameter_vectors():
    '''
    This function parses e.txt and s.txt to get the  26-dimensional multinomial
    parameter vector (characters probabilities of English and Spanish) as
    descibed in section 1.2 of the writeup

    Returns: tuple of vectors e and s
    '''
    #Implementing vectors e,s as lists (arrays) of length 26
    #with p[0] being the probability of 'A' and so on
    e=[0]*26
    s=[0]*26

    with open('e.txt',encoding='utf-8') as f:
        for line in f:
            #strip: removes the newline character
            #split: split the string on space character
            char,prob=line.strip().split(" ")
            #ord('E') gives the ASCII (integer) value of character 'E'
            #we then subtract it from 'A' to give array index
            #This way 'A' gets index 0 and 'Z' gets index 25.
            e[ord(char)-ord('A')]=float(prob)
    f.close()

    with open('s.txt',encoding='utf-8') as f:
        for line in f:
            char,prob=line.strip().split(" ")
            s[ord(char)-ord('A')]=float(prob)
    f.close()

    return (e,s)


def shred(filename):
    #Using a dictionary here. You may change this to any data structure of
    #your choice such as lists (X=[]) etc. for the assignment
    X=dict()
    with open (filename,encoding='utf-8') as f:
        # TODO: add your code here
        line = f.read()
        
        word = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for chr in word:
            X[chr] = 0

        capLine = line.upper()
        for chr in capLine:
            if chr in X:
                X[chr] += 1
        
    return X



# TODO: add your code here for the assignment
#Q1
print("Q1")
file = shred("letter.txt")
for key,value in file.items():
    print(key, value)
#Q2
print("Q2")
X1 = file['A']
e1 = get_parameter_vectors()[0][0]
s1 = get_parameter_vectors()[1][0]
calE = X1*math.log(e1)
calS = X1*math.log(s1)
print(format(calE,"0.4f"))
print(format(calS,"0.4f"))

#Q3
print("Q3")
calSigma_E = 0
for i in range(0,26):
    calSigma_E += list(file.values())[i]*math.log(get_parameter_vectors()[0][i])

f_English = math.log(0.6)+calSigma_E
print(format(f_English,"0.4f"))

calSigma_S = 0
for t in range(0,26):
    calSigma_S += list(file.values())[t]*math.log(get_parameter_vectors()[1][t])

f_Spanish = math.log(0.4)+calSigma_S
print(format(f_Spanish,"0.4f"))

#Q4
print("Q4")
p_BayesEng = 0
if (f_Spanish-f_English <= -100):
    p_BayesEng = 1
elif (f_Spanish-f_English >= 100):
    p_BayesEng = 0
else:
    p_BayesEng = 1/(1+math.exp(f_Spanish-f_English))

print(format(p_BayesEng,"0.4f"))

# You are free to implement it as you wish!
# Happy Coding!
