
from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

""" load the dataset from a provided .npy file, re-center it around the origin and return it as a NumPy array of floats"""
def load_and_center_dataset(filename):
    x = np.load(filename)
    y = np.mean(x, axis =0)
    x = x - y
    return x

""" calculate and return the covariance matrix of the dataset as a NumPy matrix (d x d array) """
def get_covariance(dataset):
    s = dataset
    n = len(s)
    s = np.dot(np.transpose(s), s)
    s = s/(n-1)
    return s

"""  perform eigen decomposition on the covariance matrix S and return a diagonal matrix (NumPy array) with the largest m eigenvalues on the diagonal, 
and a matrix (NumPy array) with the corresponding eigenvectors as columns. """
def get_eig(S, m):
    # w is the eigenvalues 
    # v is the eigenvactor
    n = len(S)
    w, v = eigh(S, subset_by_index=[n-m, n-1])
    w[::-1].sort() 
    v = np.flip(v, axis =1)
    
    mat = []
    for i in range(m):
        mat.append([])
        for k in range(m):
            mat[i].append(0)

    for i in range(m):
        mat[i][i] = w[i]
        
    w = np.array(mat)
    
    
    
    return w,v

          
 """Return all eigenvalues and corresponding eigenvectors in similar format as get_eig that explain more than perc % of variance. """
def get_eig_perc(S, perc):
    w, v = eigh(S)
    sum = np.sum(w)
    sum = sum*perc
    w, v = eigh(S, subset_by_value=[sum, np.inf])
    w[::-1].sort()
    v = np.flip(v, axis =1)
    
    mat = []
    for i in range(len(w)):
        mat.append([])
        for k in range(len(w)):
            mat[i].append(0)

    for i in range(len(w)):
        mat[i][i] = w[i]
        
    w = np.array(mat)
    
    return w,v
            
        
"""  Project each image into a m-dimensional space and return the new representation as a d x 1 NumPy array."""
def project_image(img, U):
    alpha = np.dot(img, U)
    x_proj = np.dot(U, alpha)
    
    return x_proj
    

""" Use matplotlib to display a visual representation of the original image and the projected image side-by-side. """
def display_image(orig, proj):
    orig = np.transpose(np.reshape(orig, [32,32]))
    proj = np.transpose(np.reshape(proj, [32,32]))
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize = (9,3))
    ax1.set_title("Original")
    ax2.set_title("Projection")
    c = ax1.imshow(orig, aspect="equal")
    d = ax2.imshow(proj, aspect ="equal")
    f.colorbar(c, ax = ax1)
    f.colorbar(d, ax = ax2)
    plt.show()

