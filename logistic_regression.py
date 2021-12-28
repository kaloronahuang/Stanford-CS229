# logistic_regression.py
import numpy as np
import math
import csv

def g(z):
    return 1.0 / (1.0 + math.exp(-z))

def hypothesis(theta : np.ndarray, feature : np.ndarray):
    return g(np.matmul(theta.T, feature))

filename = 'breast-cancer-wisconsin.csv'
dataset = []
setsiz = 0
dimension = 10
trX = []
trY = []

with open(filename, 'rt') as csvfile:
    file = csv.reader(csvfile)
    for ln in file:
        datavec = [1]
        qs_flag = False
        for j in range(1, dimension):
            if ln[j] == '?':
                qs_flag = True
                break
            datavec.append(float(ln[j]))
        if qs_flag:
            continue
        setsiz += 1
        vec = np.array(datavec, dtype=np.float64)

        yistr = ln[dimension]
        yi = 0.0
        if yistr == '2':
            # benign;
            yi = np.float64(0.0)
        else:
            # malignant;
            yi = np.float64(1.0)

        trY.append(yi)
        dataset.append([np.transpose(vec), yi])
        trX.append(np.transpose(vec))
    
trX = np.array(trX)
trY = np.transpose(trY)

theta = np.transpose(np.array([0.0] * dimension))

data_id = 0
alpha = 0.075
iter_id = 0
iter_limit = 10000

prev_error = np.infty
prev_theta = theta

# theta = np.array([-0.02261108, -0.01374739,  0.03398621,  0.03003378,  0.0201744,
#        -0.00850552,  0.04588798,  0.00026049,  0.02673233, -0.00536953])

while iter_id < iter_limit:
    iter_id += 1
    # calculate the relative error;
    res_vec = np.matmul(trX, theta)
    for j in range(0, setsiz):
        res_vec[j] = g(res_vec[j])
    error_vec = res_vec - trY
    err = float(np.matmul(np.transpose(error_vec), error_vec)) / float(setsiz)
    print('Iteration {0} : {1}'.format(iter_id, err))

    xi = np.transpose(trX[data_id])  
    theta = theta + alpha * (trY[data_id] - hypothesis(theta, xi)) * xi

    data_id += 1
    data_id %= setsiz

print(theta)

print("Test Phase:")
threshold = 0.5
correct_cases = 0

for x in dataset:
    res = g(np.matmul(np.transpose(theta), x[0]))
    ans = 0
    if res <= threshold:
        ans = 0
    else:
        ans = 1
    print("Expected : {0}, Estimated : {1} {2}".format(int(x[1]), res, ans))
    if (int(x[1]) == ans):
        correct_cases += 1

print("Overall correct rate : {0}".format(1.0 * correct_cases / setsiz))