# k-means.py
import numpy as np
import random
import csv

dataset = []
color = []
dim = 2
def load_data():
    with open('data.csv', 'rt') as csvfile:
        rd = csv.reader(csvfile)
        for row in rd:
            x, y = eval(row[0]), eval(row[1])
            dataset.append(np.array([[x], [y]]))
            color.append(0)

def find_clusters(k):
    random.shuffle(dataset)
    centroids = dataset[:k]
    epoch_limit = 10000
    current_epoch = 0
    amount = len(dataset)
    while current_epoch < epoch_limit:
        for i in range(0, amount):
            dists = [np.matmul((dataset[i] - centroids[j]).T, dataset[i] - centroids[j])[0][0] for j in range(0, k)]
            color[i] = np.argmin(dists)
        for i in range(0, k):
            vec_acc = sum([dataset[j] for j in range(0, amount) if color[j] == i])
            vec_acc /= sum([1 for j in range(0, amount) if color[j] == i])
            centroids[i] = vec_acc
        current_epoch += 1
    return centroids

if __name__ == '__main__':
    load_data()
    centroids = find_clusters(2)
    print(centroids)