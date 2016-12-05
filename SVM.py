#coding=utf-8
from sklearn import svm
from random import randint
import sys

'''
#demo
X = [[0, 0], [1, 1]]
y = [0, 1]
clf = svm.SVC()
clf.fit(X, y)  

print clf.predict([[2., 2.]])
'''

print "usage: python SVM.py trainfile testfile "


def readfile(fname):
    weathers = ['snow', 'cloudy', 'sunny']
    feature = []
    label = []

    f = open(fname)
    while 1:
        line = f.readline()
        if not line:
            break
        value = line.strip().split(' ')
        for i in xrange(len(weathers)):
            if value[0].find(weathers[i]) > -1:          
                label.append(i)
                value.pop(0)
                feature.append(map(float, value))
                break
    return (feature, label)



trainfile = sys.argv[1]
testfile = sys.argv[2]

#data prepare
features = [[],[]]
labels = [[],[]]

features[0], labels[0] = readfile(trainfile)
features[1], labels[1] = readfile(testfile)

#svm init    
clf = svm.SVC()
clf.fit(features[0], labels[0])

res = clf.predict(features[1])

ac = 0.0
for i in xrange(len(res)):
    if res[i] == labels[1][i]:
        ac += 1

al = [features[0], features[1], labels[0], labels[1], res]
print map(len, al)
print ac / len(res)





