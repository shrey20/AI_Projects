import numpy as np
import copy
import heapq

#Goal state
goal = np.array([[1,2,3], [4,5,6], [7,8,0]]) 
# Queue of next possible states
pq = []
# Visited states
closed = []

""" Calculates  and returns heuristic based on manhattan distance of each tile from it's goal position."""
def heuristic(state):
    a = np.array(state[0:3])
    b = np.array(state[3:6])
    c = np.array(state[6:9])
    x = np.vstack((a,b,c))
    dist = 0
    
    for i in range(1,9):
        for row in range(len(goal)):
            for col in range(len(goal[0])):
                if (goal[row][col] == i):
                    x_goal = row
                    y_goal = col
                
                if (x[row][col] == i):
                    x_val = row
                    y_val = col
        # Calculate the manhattan distance for each tile.
        distance = abs(x_val - x_goal) + abs(y_val - y_goal)
        dist += abs(x_val - x_goal) + abs(y_val - y_goal)
    return dist

  
""" Generates and returns a list of all possible next-states from a given state.""" 
def gen_succ(state):
    a = np.array(state[0:3])
    b = np.array(state[3:6])
    c = np.array(state[6:9])
    x = np.vstack((a,b,c))
    possible = []
    
    for row in range(len(goal)):
        for col in range(len(goal[0])):
            if (x[row][col] == 0):
                x_val = row
                y_val = col
    
    if (x_val -1 > -1 ):   #up
        succ = copy.deepcopy(x)
        temp = succ[x_val -1][y_val]
        succ[x_val -1][y_val] = succ[x_val][y_val]
        succ[x_val][y_val] = temp
        lst = list(succ[0]) + list(succ[1]) + list(succ[2])
        possible.append(lst)
    
    if (x_val +1 < 3): #down
        succ = copy.deepcopy(x)
        temp = succ[x_val +1][y_val]
        succ[x_val +1][y_val] = succ[x_val][y_val]
        succ[x_val][y_val] = temp
        lst = list(succ[0]) + list(succ[1]) + list(succ[2])
        possible.append(lst)
        
    if (y_val -1 > -1 ):   #left
        succ = copy.deepcopy(x)
        temp = succ[x_val][y_val -1]
        succ[x_val][y_val -1] = succ[x_val][y_val]
        succ[x_val][y_val] = temp
        lst = list(succ[0]) + list(succ[1]) + list(succ[2])
        possible.append(lst)
    
    if (y_val +1 < 3 ):   #right
        succ = copy.deepcopy(x)
        temp = succ[x_val][y_val +1]
        succ[x_val][y_val +1] = succ[x_val][y_val]
        succ[x_val][y_val] = temp
        lst = list(succ[0]) + list(succ[1]) + list(succ[2])
        possible.append(lst)
    
    return possible

""" Print the succesor states."""
def print_succ(state):
    poss = sorted(gen_succ(state))
    
    for eachstate in poss:
        h = heuristic(eachstate)
        print(eachstate, end = " ")
        print('h={0:d}'.format(h))


""" Uses A* search algorithm to choose best next state to solve the puzzle. """
def solve(state):
    lst = list(goal[0]) + list(goal[1]) + list(goal[2])
    g = 0
    #pointer for backtracking 
    parent_index = -1
    h = heuristic(state)
    # maintain a priority queue with defining #of moves + heauristic as the priority
    heapq.heappush(pq, (g+h, copy.deepcopy(state), (g,h,parent_index)))
    
    #while there are states to explore
    while len(pq) !=0:
        traverse = heapq.heappop(pq)
        g = traverse[2][0]
        closed.append(traverse)
        
        #break if goal state is found
        if traverse[1] == lst:
            break
        
        parent_index = len(closed) -1

        #generate successive states
        poss = sorted(gen_succ(traverse[1]))
        
        #traverse the successor states
        for eachstate in poss:
            h = heuristic(eachstate)
            visit = 0
            
            #Check if the state is already visited.
            for a in closed:
                if (a[1] == eachstate):
                    visit = 1
                    break
            #Push the state in pq if not visited.
            if visit == 0:
                heapq.heappush(pq, (g+h+1, eachstate, (g+1,h,parent_index)))
        
        
    path = []
    end = copy.deepcopy(traverse)
    
    #Backtrack and print the path to the goal state.
    while True:
        path.append(end)
        x = end[2][2]
        if x == -1:
            break
        else:
            end = closed[x]

    moves = 0
    y = len(path) -1
    
    while y >=0:
        print(path[y][1], end=" ")
        print('h={0:d} moves= {1:d}'.format(path[y][2][1], moves))
        y -=1
        moves +=1
  
