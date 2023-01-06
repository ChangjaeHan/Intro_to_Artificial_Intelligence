#Author: Changjae Han
#Class: CS540
#Date: Nov 29, 2022

import heapq as hp
import numpy as np
import copy as cp


def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    
    #divide it as 3 rows to calculate easily
    fromT = [from_state[i:i+3] for i in range(0, len(from_state), 3)]
    toT = [to_state[i:i+3] for i in range(0, len(to_state), 3)]
    
    #np array to calculate easily
    fromArray = np.array(fromT)
    toArray = np.array(toT)
    
    distance = 0

    #get manhattan distance
    for i in range(1,8):
        fromIndex = np.where(fromArray == i)
        toIndex = np.where(toArray == i)
    
        distance += abs(fromIndex[0][0]-toIndex[0][0])+abs(fromIndex[1][0]-toIndex[1][0])  
    
    return distance


def print_succ(state):
    
    succ_states = get_succ(state)
    
    #print with specific format
    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))
        
    
def get_succ(state):

    succ_state = []
    npState = np.array([state[i:i+3] for i in range(0, len(state), 3)])

    #copy to save state
    temp_state = cp.deepcopy(npState)

    #zero's index
    zeroX1 = np.where(temp_state == 0)[0][0]
    zeroY1 = np.where(temp_state == 0)[1][0]
    zeroX2 = np.where(temp_state == 0)[0][1]
    zeroY2 = np.where(temp_state == 0)[1][1]


    """
    move all possible north, east, west, south 
    but fail if it goes somewhere out of range
    """
    if(zeroY1-1 >= 0 and temp_state[zeroX1][zeroY1-1] != 0):
        
        addNum = temp_state[zeroX1][zeroY1-1]
        temp_state[zeroX1][zeroY1-1] = 0
        temp_state[zeroX1][zeroY1] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)
    temp_state = cp.deepcopy(npState)
    if(zeroY1+1 <= 2 and temp_state[zeroX1][zeroY1+1] != 0):
        
        addNum = temp_state[zeroX1][zeroY1+1]
        temp_state[zeroX1][zeroY1+1] = 0
        temp_state[zeroX1][zeroY1] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)
    temp_state = cp.deepcopy(npState)
    if(zeroX1-1 >= 0 and temp_state[zeroX1-1][zeroY1] != 0):
        
        addNum = temp_state[zeroX1-1][zeroY1]
        temp_state[zeroX1-1][zeroY1] = 0
        temp_state[zeroX1][zeroY1] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)
    temp_state = cp.deepcopy(npState)
    if(zeroX1+1 <= 2 and temp_state[zeroX1+1][zeroY1] != 0):
        
        addNum = temp_state[zeroX1+1][zeroY1]
        temp_state[zeroX1+1][zeroY1] = 0
        temp_state[zeroX1][zeroY1] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)
    
    
    temp_state = cp.deepcopy(npState)
    if(zeroY2-1 >= 0 and temp_state[zeroX2][zeroY2-1] != 0):
        
        addNum = temp_state[zeroX2][zeroY2-1]
        temp_state[zeroX2][zeroY2-1] = 0
        temp_state[zeroX2][zeroY2] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)
    temp_state = cp.deepcopy(npState)
    if(zeroY2+1 <= 2 and temp_state[zeroX2][zeroY2+1] != 0):
         
        addNum = temp_state[zeroX2][zeroY2+1]
        temp_state[zeroX2][zeroY2+1] = 0
        temp_state[zeroX2][zeroY2] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)
    temp_state = cp.deepcopy(npState)
    if(zeroX2-1 >= 0 and temp_state[zeroX2-1][zeroY2] != 0):
         
        addNum = temp_state[zeroX2-1][zeroY2]
        temp_state[zeroX2-1][zeroY2] = 0
        temp_state[zeroX2][zeroY2] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)
    temp_state = cp.deepcopy(npState)
    if(zeroX2+1 <= 2 and temp_state[zeroX2+1][zeroY2] != 0):
         
        addNum = temp_state[zeroX2+1][zeroY2]
        temp_state[zeroX2+1][zeroY2] = 0
        temp_state[zeroX2][zeroY2] = addNum
        sumList = temp_state.tolist()[0]+temp_state.tolist()[1]+temp_state.tolist()[2]
        succ_state.append(sumList)

    return sorted(succ_state)
    pass


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
  
    pq = [] #open
    close = [] #close
      
    initState = state; #init state
    iterState = state; #iter state to iterate state
    parentIndex = -1 #parent
    ID = 0 #ID clarification number
    maxlen = 1 #max length of priority queue
    hp.heappush(pq,(get_manhattan_distance(initState), initState, (0, get_manhattan_distance(initState), parentIndex,ID)))

    while len(pq): #check if queue is empty

        #pop lowest f(n)
        popped = hp.heappop(pq)

        #put in close
        close.append(popped)
        
        #update state
        iterState = popped[1]      

        if iterState == goal_state:
            
            mov = 0 #count move
            goalIndex = find_Index(close, iterState)
            findNext = close[goalIndex][2][2] #iterate through (backtracking)
            output = [] #save output
            output.append(close[goalIndex])

            for elem in reversed(close): #reversed for backtracking
                
                if(elem[2][3] == findNext):
                    
                    output.append(elem)
                    findNext = elem[2][2]
                
                if(findNext == -1): #stop when finding init node
                    break
                
            for out in reversed(output): #reversed again to arrange output
                print(out[1], "h={}" .format(out[2][1]), "moves:", mov )
                mov = mov + 1
                
            print("Max queue length:", maxlen)
            return
        
        #if not already in close -> calculate g,f and put in open
        #if already in close -> do nothing
        for succ in get_succ(iterState):
            
            parent = get_ID(close,iterState)    
            
            if find_Node(close,succ) == 0:
                ID = ID + 1
                hp.heappush(pq,((popped[2][0]+1+get_manhattan_distance(succ)), succ, (popped[2][0]+1, get_manhattan_distance(succ), parent,ID)))
                

        if len(pq) >= maxlen:
            maxlen = len(pq)
        
        
    #if pq is empty ->fail
    print("pq is empty, fail") 
    exit(1)

#find index of pq by checking array
def find_Index(pq, array):
    for index in range(len(pq)):
        if pq[index][1] == array:
            return index
    return -1           
               
#find node of pq by checking array      
def find_Node(pq, array):
    for index in range(len(pq)):
        if pq[index][1] == array:
            return 1           
    return 0
  
#get ID of pq by checking array
def get_ID(pq, array):
    for index in range(len(pq)):
        if pq[index][1] == array:
            id = pq[index][2][3] 
            return id
    return 0

    

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    #print_succ([0, 4, 6, 3, 0, 1, 7, 2, 5])
    print()

    #print(get_manhattan_distance([2,1,0,4,5,6,7,0,3], [2,1,6,4,5,0,7,0,3]))
    print()
    
    
    solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print()

    
