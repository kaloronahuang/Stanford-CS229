# export_dictionary.py
import csv

gdict = {}

def processEmail(email : str):
    email = email.lower()
    elist = list(email)
    for i in range(len(elist)):
        if not ('a' <= elist[i] and elist[i] <= 'z'):
            elist[i] = ' '
    email = ''.join(elist)
    dlist = email.split(' ')
    for word in dlist:
        if word not in gdict:
            gdict[word] = 0
        gdict[word] += 1

def exportDictionary():
    with open('dict.csv', 'w') as csvfile:
        file = csv.writer(csvfile)
        sdict = []
        for word, cnt in gdict.items():
            sdict.append([cnt, word])
        sdict.sort()
        for word, cnt in sdict:
            file.writerow([word, cnt])

if __name__ == '__main__':
    with open('emails.csv', 'rt') as csvfile:
        file = csv.reader(csvfile)
        for email, attr in file:
            y_vec = eval(attr)
            processEmail(email[8::])
    
    exportDictionary()
