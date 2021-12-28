# softmax.py
import numpy as np
import struct
import math
import matplotlib.pyplot as plt

train_image_path = 'train-images-idx3-ubyte'
train_label_path = 'train-labels-idx1-ubyte'
test_image_path = 't10k-images-idx3-ubyte'
test_label_path = 't10k-labels-idx1-ubyte'

# data = [np.ndarry, int]
images = []
labels = []
amount = 60000
dataset = []
row, col = 28, 28
x_dimension = row * col + 1
y_dimension = 10

def hypothesis(theta : np.ndarray, feature : np.ndarray):
    res = np.zeros(shape=[y_dimension], dtype=np.float128)
    for i in range(0, y_dimension):
        res[i] = np.matmul(theta[i], feature)
    max_idx = res.max()
    exp_sum = 0.0
    for i in range(0, y_dimension):
        res[i] -= max_idx
        res[i] = math.exp(res[i])
        exp_sum += res[i]
    for i in range(0, y_dimension):
        res[i] = res[i] / exp_sum
    return res

def predict(theta : np.ndarray, feature : np.ndarray):
    hypo = hypothesis(theta, feature)
    term_id = 0
    for i in range(1, y_dimension):
        if hypo[term_id] < hypo[i]:
            term_id = i
    return [term_id, hypo[term_id] / hypo.sum()]

def gradient_descent(alpha : np.float128, epoch : int):
    theta = np.zeros([y_dimension, x_dimension])
    for idx in range(epoch):
        correct_cases = 0
        for i in range(len(dataset)):
            num, prob = predict(theta, dataset[i][0])
            if num == labels[i]:
                correct_cases += 1
        print('Epoch {0}|Success rate: {1}%'.format(idx, 100.0 * correct_cases / len(dataset)))
        for data in dataset:
            img_vec, label_vec = data[0], data[1]
            predicted = hypothesis(theta, img_vec) - label_vec
            
            for i in range(0, y_dimension):
                err = predicted[i] - label_vec[i]
                grad = err * img_vec
                theta[i] = theta[i] - alpha * grad
    return theta

def load_train_data():
    image_file = open(train_image_path, 'rb')
    label_file = open(train_label_path, 'rb')
    image_file.read(16)
    label_file.read(8)

    for idx in range(0, amount):
        img, trimg = [], [1.0]
        label = struct.unpack('B', label_file.read(1))[0]
        for i in range(0, row):
            ln = []
            for j in range(0, col):
                pixel = struct.unpack('B', image_file.read(1))[0]
                ln.append(pixel), trimg.append(np.float128(float(pixel) / 255.0))
            img.append(ln)
        label_vec = np.zeros(shape=[y_dimension])
        label_vec[label] = 1
        images.append(img), dataset.append([np.array(trimg, dtype=np.float128), np.array(label_vec, dtype=np.float128)])
        labels.append(label)

    image_file.close()
    label_file.close()

def run_test(theta : np.ndarray):
    image_file = open(test_image_path, 'rb')
    label_file = open(test_label_path, 'rb')
    image_file.read(16)
    label_file.read(8)
    test_amount = 10000
    correct_cases = 0
    for idx in range(0, test_amount):
        trimg = [1.0]
        label = struct.unpack('B', label_file.read(1))[0]
        for i in range(0, row):
            for j in range(0, col):
                pixel = struct.unpack('B', image_file.read(1))[0]
                trimg.append(np.float128(float(pixel) / 255.0))
        res, prob = predict(theta, trimg)
        if res == label:
            correct_cases += 1
    print('Test has been run through; the success rate is : {0}%'.format(100.0 * correct_cases / test_amount))
    image_file.close()
    label_file.close()

def load_model(filename : str):
    return np.load(filename)

def save_model(theta : np.ndarray, filename : str):
    np.save(filename, theta)
    return

def run():
    load_train_data()
    Gtheta = gradient_descent(0.05, 10)
    # Gtheta = load_model('model.bin')
    save_model(Gtheta, 'model.bin')
    run_test(Gtheta)

if __name__ == '__main__':
    run()