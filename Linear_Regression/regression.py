import numpy as np
from matplotlib import pyplot as plt
import csv
import random

""" Takes  a string representing the path to the csv file and returns an n by m+1 array, where n is # data points and m is #features.
The labels y are in the first column"""
def get_dataset(filename):
    dataset = None
    dataset = []
    with open(filename) as csvfile:
        
            dataset = list(csv.reader(csvfile))
    dataset = np.asarray(dataset)
    dataset = np.delete(dataset, 0, 0)
    dataset = np.delete(dataset, 0, 1)
    dataset = np.asarray(dataset, dtype = float)
    return dataset


""" Prints the mean and the variance for the specified columns in the given dataset."""
def print_stats(dataset, col):
    n = len(dataset)
    print(n)
    
    column = dataset[:, col]
    mean = 0
    for i in column:
        mean = mean + i
    mean = mean/n
    print('{0:.2f}'.format(mean))

    std = 0
    var  = sum(pow(x-mean,2) for x in column) / (n-1)  # variance
    std  = np.sqrt(var)  # standard deviation
    print('{0:.2f}'.format(std))

    
    pass


"""Calculates and returns the mean squared error on the dataset given fixed betas."""
def regression(dataset, cols, betas):
    mse = None
    column = dataset[:, cols]

    n = len(column)

    mse = np.sum(np.square(np.dot(column, betas[1:]) + betas[0] - dataset[:,0]))/n
    return mse

""" Performs a single step of gradient descent on the MSE and returns the derivative values as an 1D array."""
def gradient_descent(dataset, cols, betas):
    grads = None
    column = dataset[:,cols]
    n = len(column)
    x = np.dot(column, betas[1:]) + betas[0]-dataset[:,0]
    mse = 2*np.sum(x)/n
    m = len(betas)
    grads = np.zeros(m)
    grads[0] = mse
    for i in range(m-1):
        grads[i+1] = 2*np.sum(x*column[:,i])/n
    return grads


""" Performs T iterations of gradient descent starting at the given betas and prints the results; does not return anything."""
def iterate_gradient(dataset, cols, betas, T, eta):
    n = len(betas)
    for i in range(T):
        grads = gradient_descent(dataset,cols,betas)
        for j in range(len(betas)):
            betas[j] = betas[j] - eta*grads[j]
        mse = regression(dataset, cols, betas)
        
        print(i+1, end=' ')
        print('{0:.2f}'.format(mse), end = ' ')
        for x in betas:
            print('{0:.2f}'.format(x), end = ' ')
        print()
    pass


"""  Using the closed-form solution, calculates and returns the values of betas and the corresponding MSE as a tuple. """
def compute_betas(dataset, cols):
    betas = None
    mse = None
    column = dataset[:,cols]
    n = len(column)
    bias = np.zeros(n)
    for i in range(n):
        bias[i] = 1
    bias=bias[:,None]

    column=np.concatenate((bias,column),1)
 
    X = column
    y = np.transpose(dataset[:,0])
 
    betas=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
   
    mse = regression(dataset, cols, betas)
    return (mse, *betas)

""" Using the closed-form solution betas, return the predicted body fat percentage of the give features."""
def predict(dataset, cols, features):
    result = None
    
    bet = compute_betas(dataset,cols)
    bet = np.asarray(bet)
    bet = np.delete(bet, 0)
    features = np.append([1], features)
    result = np.dot(features, bet)
    
    return result


"""generates two synthetic datasets, one using a linear model and the other using a quadratic model."""
def synthetic_datasets(betas, alphas, X, sigma):
    n = len(X)


    Z = np.random.normal(loc = 0.0, scale= sigma, size =n)
    X = np.hstack((np.ones((len(X), 1)), X))
    linears = X @ betas + Z
    
    
    linears = np.hstack((linears.reshape((-1, 1)), X[:, 1:]))
    linear = np.hstack((np.ones((len(X), 1)), X))
    
    Z1 = np.random.normal(loc = 0.0, scale= sigma, size =n)
 
    p = np.hstack((X[:, 0].reshape((-1, 1)), (X[:, 1] ** 2).reshape(-1, 1)))
    quads = p @ alphas + Z1
    quads = np.hstack((quads.reshape((-1, 1)), X[:, 1:]))

    
    return linears, quads

"""fits the synthetic datasets, and plots a figure depicting the MSEs under different situations."""
def plot_mse():
    from sys import argv
    if len(argv) == 2 and argv[1] == 'csl':
        import matplotlib
        matplotlib.use('Agg')
        x = np.linspace(-100.0, 100.0, 1000).reshape(-1,1)
   
        betas = np.array([1,2])

        alphas = np.array([-1,1])
    
        sigma = 10**-4
        sig_list = []
        while(sigma< 10**6):
            
            sig_list.append(sigma)
            sigma = sigma*10
        
 
        lin = []
        quad = []
        for i in sig_list:
            a,b = synthetic_datasets(betas, alphas, x, i)
            lin.append(compute_betas(a, cols=[1])[0])
            quad.append(compute_betas(b, cols=[1])[0])

        plt.plot(sig_list,lin, '-o', label = 'linear')
        plt.plot(sig_list,quad, '-o', label = 'quadratic')
        plt.legend(loc='upper left')
        plt.xlabel('Sigma')
        plt.ylabel('MSE')
        plt.yscale("log")
        plt.xscale("log")

        plt.savefig('mse.pdf')


if __name__ == '__main__':
    plot_mse()
