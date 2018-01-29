# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:07:16 2018

@author: wel12
"""

import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def readData(location):
    h = 'carEvaluation\\attributes.csv'
    a = read_csv(h, header=None)
    #temp = a[a.columns[0]]
    #headers = temp.values
    #headers = np.append(headers, '')
    dataset = read_csv(location, header=None,  na_values=" ?") #, names=headers
    return dataset

def numberfyData(raw):
    keys = []
    out = raw
    for column in raw:
        keys.append(set(raw[column]))
        out[column] = raw[column].replace(set(raw[column]),range(len(set(raw[column]))))
    return out, keys


def testModle(data_train, data_test, targets_train, targets_test, classifier):
    modle = classifier.fit(data_train, targets_train)
    return modle.predict(data_test)

def displayTest(targets_predicted,targets_test):
    simCount = 0
    difCount = 0
    print ("predicted")
    print (targets_predicted)
    print("test")
    test = np.array(targets_test)
    print(test)
    for i in range(len(targets_predicted)):
        print (targets_predicted[i], "==", test[i])
        #simCount +=1
        if targets_predicted[i] == test[i]:
            simCount += 1
        if targets_predicted[i] != test[i]:
            difCount += 1
    print("\tsim =\t", simCount)
    print("\tdif =\t", difCount)
    print("\ttotal =\t", len(targets_predicted))
    print("\t\t%",simCount/len(targets_predicted) * 100)
    return simCount/len(targets_predicted) * 100


class KNNClasifier:
    def __init__(self):
        pass
    
    def fit(self, data_train, targets_train):
        return KNNModle(data_train, targets_train)
    
def eDistance(a, b):
    out = 0
    if not(np.isscalar(a)):
        r = len(a)
        for point in range(r):
            out = out + (int(a[point])-int(b[point]))**2
    else:
        out = (a-b)**2
    return out
    
    
class KNNModle:
    def __init__(self,data_train, targets_train):
        self.data_train    = data_train
        self.targets_train = targets_train
        self.k = 3
        return None
    
    def getNebors(self, data_test_row):
        nebors = []
        distances = []
        #find how far everything is from eachother
        test = np.array(self.data_train)
        for data_train_row in test:
            #distances.append(np.asscalar(LA.norm(data_train_row - data_test_row)))  #Euclidean distance: LA.norm(row - data_train_row
            distances.append(eDistance(data_train_row, data_test_row))
        #combine rlivent data
        nebors = list(zip(distances, self.targets_train, self.data_train))
        
        #sort based on distance
        nebors = sorted(nebors, key=lambda k:k[0])          
        # cut so their are only k number # default is 3
        nebors = nebors[:self.k]     
        return nebors
    
    def determinSimilarity(self, nebors):
        # get list of top targets
        possabilities = list(zip(*nebors))[1]
        #find unique values of top targets
        targetRange = set(possabilities)
        count = []
        #loop through values of top targets and count how many there are 
        for num in targetRange:
            count.append(list(possabilities).count(num))
        # combine values and counts
        out = list(zip(targetRange, count))
        # sort
        out = sorted(out, key=lambda k:k[1], reverse=True)
        # return 
        return out[0][0]
    
    def predict(self, data_test):
        targets = []
        test = np.array(data_test)
        for data_test_row in test:
            nebors = self.getNebors(data_test_row)
            targets.append(self.determinSimilarity(nebors))
        return targets





def main(argv):
    
    rawData = readData('carEvaluation\car.data')
    numData, key = numberfyData(rawData)
    target = numData.iloc[:,-1]
    
    print ("DATA :")
    print (numData)
    print ("TARGETS:")
    print (target)
    data_train, data_test, targets_train, targets_test = train_test_split(numData, target, test_size = .3)
    print ("DATA Train:")
    print (data_train)
    print ("DATA Test:")
    print (data_test)
    print ("targts Train:")
    print (targets_train)
    print ("targets test:")
    print (targets_test)
    
    
    print("My KNN")
    classifier = KNNClasifier()
    targets_predicted = testModle(data_train, data_test, targets_train, targets_test, classifier)
    displayTest(targets_predicted,targets_test)
    return None




if __name__ == "__main__":
    main(sys.argv)