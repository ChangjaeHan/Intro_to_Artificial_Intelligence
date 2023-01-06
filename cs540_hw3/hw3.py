#Author: ChangjaeHan

from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt


def load_and_center_dataset(filename):
    # Your implementation goes here!
    x = np.load(filename)
    center = x-np.mean(x, axis=0)
    return center


def get_covariance(dataset):
    # Your implementation goes here!
    cov = (1/2413)*np.dot(np.transpose(dataset),dataset)
    return cov


def get_eig(S, m):
    # Your implementation goes here!
    w,v = eigh(S, subset_by_index=[1024-m, 1023])
    w[[0,1]] = w[[1,0]]
    v[:,[0,1]] = v[:,[1,0]]
    return np.diag(w),v


def get_eig_prop(S, prop):
    # Your implementation goes here!   
    T = eigh(S, eigvals_only=True)
    sum = 0
    for j in range(0,1024):
        sum += T[j]
    w,v = eigh(S, subset_by_value=[prop*sum,np.inf])
    w[[0,1]] = w[[1,0]]
    v[:,[0,1]] = v[:,[1,0]]
    return np.diag(w), v
    

def project_image(image, U):
    # Your implementation goes here!
    alpha = 0
    for j in range(2):
        alpha += np.dot(np.dot(np.transpose(U[:,j]),image),U[:,j])

    return alpha


def display_image(orig, proj):
    # Your implementation goes here!
    reshapedProj = proj.reshape(32,32)
    reshapedOrig = orig.reshape(32,32)
    rotP = np.rot90(reshapedProj,3)
    rotO = np.rot90(reshapedOrig,3)
    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.set_title('Original')
    ax2.set_title('Projection')
    col1 = ax1.imshow(rotO,aspect='equal')
    col2 = ax2.imshow(rotP,aspect='equal')
    fig.colorbar(col1, ax=ax1)
    fig.colorbar(col2, ax=ax2)
    return(plt.show())




