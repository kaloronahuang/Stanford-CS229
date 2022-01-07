# bayes.py
import export_dictionary
import csv
import numpy as np

# spam : 1, ham : 0;

gdict = {}
trainingData = []
phi = []
phiy = 0

def readDictionary(filename : str):
    global phi
    with open(filename, 'rt') as csvfile:
        file = csv.reader(csvfile)
        idcnt = 0
        for word, cnt in file:
            gdict[word] = idcnt
            idcnt += 1
    phi = np.zeros([len(gdict), 2])

def loadTrainingData():
    with open('emails.csv', 'rt') as csvfile:
        file = csv.reader(csvfile)
        for content, spam in file:
            email = content[8::].lower()
            elist = list(email)
            for i in range(len(elist)):
                if not ('a' <= elist[i] and elist[i] <= 'z'):
                    elist[i] = ' '
            email = ''.join(elist)
            dlist = email.split(' ')
            xvec, y = np.zeros(len(gdict)), eval(spam)
            for word in dlist:
                if word in gdict:
                    xvec[gdict[word]] = 1
            trainingData.append([xvec, y])

def processVectors():
    global phiy
    spam_cnt = 0
    for xvec, y in trainingData:
        if y == 1:
            spam_cnt += 1
        for i in range(len(gdict)):
            phi[i][y] += xvec[i]
    for i in range(len(gdict)):
        phi[i][0] += 1
        phi[i][1] += 1
        phi[i][0] = phi[i][0] / (len(trainingData)- spam_cnt + 2)
        phi[i][1] = phi[i][1] / (spam_cnt + 2)
    phiy = spam_cnt / len(trainingData)

def analyze(xvec : np.ndarray):
    numerator0 = 0.0
    numerator1 = 0.0
    for i in range(len(gdict)):
        if xvec[i] == 1:
            numerator1 += phi[i][1] * phiy
            numerator0 += phi[i][0] * (1.0 - phiy)
        else:
            numerator1 += (1.0 - phi[i][1]) * phiy
            numerator0 += (1.0 - phi[i][0]) * (1.0 - phiy)
    return numerator1 / (numerator0 + numerator1)

def perceptron(x : float):
    if x <= 0.5:
        return 0
    else:
        return 1

if __name__ == '__main__':
    readDictionary('dict.csv')
    loadTrainingData()
    processVectors()
    correct_cases = 0
    for email, y in trainingData:
        verdict = perceptron(analyze(email))
        correct_cases += (verdict == y)
    print('Correct rate : {0}%'.format(correct_cases / len(trainingData) * 100.0))