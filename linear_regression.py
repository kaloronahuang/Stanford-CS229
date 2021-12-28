# linear_regression.py
import numpy as np
import math
import csv

from numpy.lib.npyio import load

trX = []
trY = []
dataset = []
dimension = 7

def load_data(filename : str):
    with open(filename, 'rt') as csvfile:
        file = csv.reader(csvfile)
        for ln in file:
            vecx, col = [1], len(ln)
            for i in range(1, col - 1):
                vecx.append(np.float64(ln[i]))
            y = np.float64(ln[col - 1])
            trX.append(vecx), trY.append(y)
            dataset.append([np.array(vecx), y])

def hypothesis(theta : np.ndarray, feature : np.ndarray):
    return np.matmul(theta.T, feature)

def gradient(theta : np.ndarray, feature : np.ndarray, res : np.float64):
    hypo = hypothesis(theta, feature)
    err = hypo - res
    return err * feature

def loss(theta : np.ndarray):
    err_vec = np.matmul(trX, theta) - trY
    return np.matmul(err_vec.T, err_vec)

def gradient_descent(alpha : np.float64, epoch : int):
    theta = np.array([0.0] * dimension)
    for iter in range(epoch):
        for data in dataset:
            theta = theta - alpha * gradient(theta, data[0], data[1])
            print('Iteration {0}: {1}'.format(iter, loss(theta)))

def run():
    load_data('dataset.csv')
    gradient_descent(1e-7, 1)

if __name__ == '__main__':
    run()