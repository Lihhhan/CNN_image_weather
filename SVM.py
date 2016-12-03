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


#data prepare
weathers = ['snow', 'cloudy', 'sunny']
features = [[],[]]
labels = [[],[]]

f = open(sys.argv[1])

while 1:
    line = f.readline()
    if not line:
        break
    value = line.strip().split(' ')
    for i in xrange(len(weathers)):
        if value[0].find(weathers[i]) > -1: 
            #train set and test set 
            r = randint(0, 3)
            if r == 0:
                labels[1].append(i) 
                value.pop(0)
                features[1].append(map(float, value))
            else:
                labels[0].append(i)
                value.pop(0)
                features[0].append(map(float, value))
            break
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





