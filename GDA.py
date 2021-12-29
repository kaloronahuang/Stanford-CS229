# GDA.py
from math import pi, exp
import numpy as np
import csv

from numpy.core.fromnumeric import shape

# 0 for benign, 1 for malignant;

phi = 0.0
benign = []
benign_avg = []
malignant = []
malignant_avg = []
dimension = 9
sigma = np.zeros(shape=[dimension, dimension])

def load_data(filename : str):
    with open(filename, 'rt') as csvfile:
        file = csv.reader(csvfile)
        for ln in file:
            vec, flag = [], False
            for i in range(1, dimension + 1):
                if ln[i] == '?':
                    flag = True
                    break
                vec.append(np.float64(ln[i]))
            if flag:
                continue
            verdict = (int(ln[dimension + 1]) // 2) - 1
            if verdict == 0:
                benign.append(np.array(vec))
            else:
                malignant.append(np.array(vec))

def process():
    global phi
    global sigma
    global benign_avg
    global malignant_avg
    benign_avg = sum(benign) / len(benign)
    malignant_avg = sum(malignant) / len(malignant)
    for vec in benign:
        stdvec = vec - benign_avg
        stdvec = stdvec.reshape(dimension, 1)
        sigma += np.matmul(stdvec, stdvec.T)
    for vec in malignant:
        stdvec = vec - malignant_avg
        stdvec = stdvec.reshape(dimension, 1)
        sigma += np.matmul(stdvec, stdvec.T)
    n = len(malignant) + len(benign)
    sigma /= n
    phi = len(malignant) / n

def predict(feature : np.ndarray):
    p_y_0 = 1.0 - phi
    p_y_1 = phi
    sigma_inv = np.linalg.inv(sigma)
    sigma_det = np.linalg.det(sigma)
    p_x_y_0 = 1.0 / (pow(2.0 * pi, dimension / 2) * pow(sigma_det, 0.5)) * exp(-0.5 * np.matmul(np.matmul(((feature - benign_avg).reshape(dimension, 1)).T, sigma_inv), (feature - benign_avg)))
    p_x_y_1 = 1.0 / (pow(2.0 * pi, dimension / 2) * pow(sigma_det, 0.5)) * exp(-0.5 * np.matmul(np.matmul(((feature - malignant_avg).reshape(dimension, 1)).T, sigma_inv), (feature - malignant_avg)))
    p_0 = p_y_0 * p_x_y_0
    p_1 = p_y_1 * p_x_y_1
    if p_0 > p_1:
        return 0
    else:
        return 1

def main():
    load_data('breast-cancer-wisconsin.csv')
    process()
    correct_cases = 0
    for vec in benign:
        if predict(vec) == 0:
            correct_cases += 1
    for vec in malignant:
        if predict(vec) == 1:
            correct_cases += 1
    print('Success rate : {0}%'.format(100.0 * correct_cases / (len(benign) + len(malignant))))

if __name__ == '__main__':
    main()