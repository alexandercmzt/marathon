import csv
import random
import numpy as np
from classifiers import NaiveBayes as NB

def loadCsv(filename):
    lines = csv.reader(open(filename, "rb"))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataset, splitRatio):
    '''
    splits a dataset into training and testing sets
    splitRatio is the ratio of the set that should be for training
    returns an array with training set as the 0th index and test set as the 1st
    '''
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

def test():
    filename = 'pima-indians.data.csv'
    dataset = loadCsv(filename)
    splitRatio = 0.67
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    tSet = np.array(trainingSet)
    vSet = np.array(testSet)
    featureTypes = [True, True, True, True, True, True, True, True]
    print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(tSet), len(vSet))
    nb = NB(tSet, featureTypes)
    summaries = nb.train()
    #print('Summaries: {0}').format(summaries)
    predictions = nb.predict(vSet)
    #print('Predictions: {0}').format(predictions)
    accuracy = nb.getAccuracy(vSet, predictions)
    print('Accuracy: {0}%').format(accuracy)

test()
