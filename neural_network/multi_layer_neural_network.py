# multi_layer_neural_network.py
import random
import numpy as np
import struct

train_image_path = 'train-images-idx3-ubyte'
train_label_path = 'train-labels-idx1-ubyte'
test_image_path = 't10k-images-idx3-ubyte'
test_label_path = 't10k-labels-idx1-ubyte'
row, col = 28, 28

input_dim = row * col
output_dim = 10
batch_siz = 10
dataset = []
testset = []
layers_weight = []
layers_bias = []
layers_shape = []
layers_input = []
layers_output = []
layers_delta = []
layers_num = 0

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def dsigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))

def forward_pass(xvec : np.ndarray):
    for i in range(0, layers_num):
        layers_input[i] = np.matmul(layers_weight[i], xvec) + layers_bias[i]
        xvec = layers_output[i] = sigmoid(layers_input[i])
    return xvec

def backward_pass(yvec : np.ndarray):
    layers_delta[layers_num - 1] = np.multiply((layers_output[layers_num - 1] - yvec), dsigmoid(layers_input[layers_num - 1]))
    for i in range(layers_num - 2, -1, -1):
        layers_delta[i] = np.multiply(np.matmul(np.transpose(layers_weight[i + 1]), layers_delta[i + 1]), dsigmoid(layers_input[i]))

def run_test():
    correct_cases = 0
    for (xvec, label) in testset:
        correct_cases += int(np.argmax(forward_pass(xvec)) == label)
    return correct_cases / len(testset)

def update_mini_batch(mini_batch, lr : float):
    global layers_weight
    global layers_bias
    dJ_dW = [np.zeros(w.shape) for w in layers_weight]
    dJ_dB = [np.zeros(b.shape) for b in layers_bias]
    for (xvec, yvec) in mini_batch:
        forward_pass(xvec)
        backward_pass(yvec)
        outputs = [xvec] + layers_output
        dJ_dW = [(dJ_dW[i] + np.matmul(layers_delta[i], np.transpose(outputs[i]))) for i in range(0, layers_num)]
        dJ_dB = [(dJ_dB[i] + layers_delta[i]) for i in range(0, layers_num)]
    layers_weight = [l_w - (lr / len(mini_batch)) * d_w for l_w, d_w in zip(layers_weight, dJ_dW)]
    layers_bias = [l_b - (lr / len(mini_batch)) * d_b for l_b, d_b in zip(layers_bias, dJ_dB)]

def train(epoch_limit : int, lr : float):
    epoch_id = 0
    dataset_siz = len(dataset)
    while epoch_id < epoch_limit:
        epoch_id += 1
        random.shuffle(dataset)
        mini_batches = [dataset[k:k + batch_siz] for k in range(0, dataset_siz, batch_siz)]
        for mini_batch in mini_batches:
            update_mini_batch(mini_batch, lr)
        print('[Epoch {0}]Correct rate : {1}%'.format(epoch_id, run_test() * 100.0))

def add_layer(siz : int):
    global layers_num
    layers_num += 1
    layers_shape.append(siz)
    prev_siz = input_dim
    if layers_num > 1:
        prev_siz = layers_shape[layers_num - 2]
    layers_weight.append(np.random.randn(siz, prev_siz))
    layers_bias.append(np.random.randn(siz, 1))
    layers_input.append(np.zeros((siz, 1)))
    layers_output.append(np.zeros((siz, 1)))
    layers_delta.append(np.zeros((siz, prev_siz)))

def define_model():
    add_layer(input_dim)
    add_layer(30)
    add_layer(output_dim)

def load_dataset():
    image_file = open(train_image_path, 'rb')
    label_file = open(train_label_path, 'rb')
    image_file.read(16)
    label_file.read(8)

    for idx in range(0, 60000):
        trimg = []
        label = struct.unpack('B', label_file.read(1))[0]
        for i in range(0, row):
            for j in range(0, col):
                pixel = struct.unpack('B', image_file.read(1))[0]
                trimg.append(float(float(pixel) / 255.0))
        label_vec = np.zeros(shape=[output_dim, 1])
        label_vec[label] = 1
        trimg = np.array(trimg).reshape(input_dim, 1)
        dataset.append((trimg, label_vec))

    image_file.close()
    label_file.close()

def load_testset():
    image_file = open(test_image_path, 'rb')
    label_file = open(test_label_path, 'rb')
    image_file.read(16)
    label_file.read(8)
    test_amount = 10000
    for idx in range(0, test_amount):
        trimg = []
        label = struct.unpack('B', label_file.read(1))[0]
        for i in range(0, row):
            for j in range(0, col):
                pixel = struct.unpack('B', image_file.read(1))[0]
                trimg.append(float(float(pixel) / 255.0))
        trimg = np.array(trimg).reshape(input_dim, 1)
        testset.append((trimg, label))
    image_file.close()
    label_file.close()

if __name__ == '__main__':
    load_dataset()
    load_testset()
    define_model()
    train(10, 3.0)