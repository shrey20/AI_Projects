
import os
import math
import csv
import numpy as np
import random
import matplotlib.pyplot as plt

""" takes in a string with a path to a CSV file formatted as in the link above, and returns the first 20 data points 
(without the Generation and Legendary columns but retaining all other columns) in a single structure."""
def load_data(filepath):
        data = []
        count = 0
        with open(filepath) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if count < 20:
                    data.append({'#': int(row['#']), 'Name' : row['Name'], 'Type 1': row['Type 1'], 'Type 2': row['Type 2'], 'Total':int(row['Total']) , 'HP': int(row['HP']), 'Attack': int(row['Attack']), 'Defense': int(row['Defense']), 'Sp. Atk': int(row['Sp. Atk']), 'Sp. Def': int(row['Sp. Def']), 'Speed': int(row['Speed'])})
                    count+= 1         
        return data

"""takes in one row from the data loaded from the previous function, calculates the corresponding 
x, y values for that Pokemon as specified above, and returns them in a single structure."""
def calculate_x_y(stats):
    x = stats['Attack']+ stats['Sp. Atk']+ stats['Speed']
    y = stats['Defense']+ stats['Sp. Def']+ stats['HP']
    tup1 = (x, y)
    return tup1

""" performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y)
feature representation, and returns a data structure representing the clustering. """
def hac(dataset):   
    b=0 
    while b in range(len(dataset)) :
        if dataset[b][0] == None or dataset[b][0] == np.inf or dataset[b][0] == -np.inf:
            dataset.pop(b)
            continue
        elif dataset[b][1] == None or dataset[b][1] == np.inf or dataset[b][1] == -np.inf:
            dataset.pop(b)
            continue
        else:
            b +=1
            
    m = len(dataset)
    n = m
    output = np.zeros((m-1, 4))
    min_dis = np.zeros((m,m))
    row = -1
    col = -1
    cluster = {}
    c = 0
    def get_key(val):
        for key, value in cluster.items():
            if val in value:
                return key
        return "key doesn't exist" 
    
    while(c<m):
        cluster[c] = [c]
        c += 1
    
    
    for i in range(len(dataset)):
        dataset[i] = np.array(dataset[i]) 

    for i in dataset:
        row += 1
        col = -1
        for j in dataset:
            col += 1
            dist = np.sqrt(np.sum(np.square(i - j)) )
            min_dis[row][col] = dist
            if row == col:
                min_dis[row][col] = np.inf
    

    for i in range(len(output)):
        a = min_dis.min()

        tie_brk = np.inf
        for h in range(len(min_dis)):
            for l in range(len(min_dis)):
                if min_dis[h][l] == a:
                    if h <= l and h<tie_brk:
                        tie_brk = h
                        r = h
                        c = l
                    elif l<h and l<tie_brk:
                        tie_brk = l
                        r = h
                        c = l
                    
                    
        output[i][2] = min_dis[r][c]

        r1 = get_key(r)
        c1 = get_key(c)

        if r1<=c1:
            output[i][0] = r1
            output[i][1] = c1
        else:
            output[i][0] = c1
            output[i][1] = r1
        
        
        output[i][3] = len(cluster[r1]) + len(cluster[c1])
        cluster[n] = cluster[r1]+cluster[c1]
        
        del(cluster[r1])
        del(cluster[c1])
        
        for i in range(len(cluster[n])):
            for j in range(i+1, len(cluster[n])):
                min_dis[cluster[n][i]][cluster[n][j]] = np.inf
                min_dis[cluster[n][j]][cluster[n][i]] = np.inf
        n += 1
        
        
      
        
    return output        

"""  takes in the number of samples we want to randomly generate, and returns these samples in a single structure."""
def random_x_y(m):
    list_r = []
    
    for i in range(m):
        tup1 = (int(random.uniform(1,360)),int(random.uniform(1,360)))
        list_r.append(tup1)
    
    return list_r

  
""" performs single linkage hierarchical agglomerative clustering on the Pokemon with the (x,y) feature representation, and imshow the clustering process."""
def imshow_hac(dataset):
    
    b=0 
    while b in range(len(dataset)) :
        if dataset[b][0] == None or dataset[b][0] == np.inf or dataset[b][0] == -np.inf:
            dataset.pop(b)
            continue
        elif dataset[b][1] == None or dataset[b][1] == np.inf or dataset[b][1] == -np.inf:
            dataset.pop(b)
            continue
        else:
            b +=1
            
    m = len(dataset)
    n = m
    output = np.zeros((m-1, 4))
    min_dis = np.zeros((m,m))
    row = -1
    col = -1
    cluster = {}
    c = 0
    def get_key(val):
        for key, value in cluster.items():
            if val in value:
                return key
        return "key doesn't exist" 
    
    while(c<m):
        cluster[c] = [c]
        c += 1    
    
    for i in range(len(dataset)):
        dataset[i] = np.array(dataset[i]) 
    
    
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0],dataset[i][1])
    plt.pause(0.1)

    for i in dataset:
        row += 1
        col = -1
        for j in dataset:
            col += 1
            dist = np.sqrt(np.sum(np.square(i - j)) )
            min_dis[row][col] = dist
            if row == col:
                min_dis[row][col] = np.inf
    

    for i in range(len(output)):
        a = min_dis.min()
        tie_brk = np.inf
        for h in range(len(min_dis)):
            for l in range(len(min_dis)):
                if min_dis[h][l] == a:
                    if h <= l and h<tie_brk:
                        tie_brk = h
                        r = h
                        c = l
                    elif l<h and l<tie_brk:
                        tie_brk = l
                        r = h
                        c = l
                    
                    

        plt.plot([dataset[r][0],dataset[c][0]], [dataset[r][1],dataset[c][1]])
        plt.pause(0.1)
        output[i][2] = min_dis[r][c]

        r1 = get_key(r)
        c1 = get_key(c)
        
        if r1<=c1:
            output[i][0] = r1
            output[i][1] = c1
        else:
            output[i][0] = c1
            output[i][1] = r1
        
        
        output[i][3] = len(cluster[r1]) + len(cluster[c1])
        cluster[n] = cluster[r1]+cluster[c1]
        del(cluster[r1])
        del(cluster[c1])

        for i in range(len(cluster[n])):
            for j in range(i+1, len(cluster[n])):
                min_dis[cluster[n][i]][cluster[n][j]] = np.inf
                min_dis[cluster[n][j]][cluster[n][i]] = np.inf
        
        n += 1
        
        
      
        
    plt.show()           
        




