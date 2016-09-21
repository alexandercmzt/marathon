# Implementation of a Naive Bayes classifier
# This classifier can accept both continuous and discrete inputs.
# It assumes Guassian distributions for continuous inputs.
# Based on the tutorial at: http://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
import sys
import csv
import random
import math
import types

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

def separateByClass(dataset):
    '''
    returns a map of class to all points in dataset that fall into that class
    e.g. {0: [[2, 21, 0]], 1: [[1, 20, 1], [3, 22, 1]]}
    '''
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def getClassProbabilities(dataset):
    '''
    returns a map of class to its probability, as computed from dataset
    '''
    probabilities = {}
    dataSetSize = len(dataset)
    separated = separateByClass(dataset)
    for classValue, instances in separated.iteritems():
        probabilities[classValue] = float(len(instances))/dataSetSize
    return probabilities

def getDiscreteProbabilities(numbers):
    '''
    here, numbers is a list of discrete values
    this function returns a map of each discrete values to its probability within the list
    '''
    setSize = len(numbers)
    counts = {}
    for value in numbers:
        if value not in counts:
            counts[value] = 0
        counts[value] += 1
    probabilities = {}
    for value, count in counts.iteritems():
        probabilities[value] = float(count) / setSize
    return probabilities

def summarize(featureTypes, dataset):
    '''
    featureTypes is an array indicating true at index i if feature i is continuous, and false
    if feature i is discrete
    returns an array of:
      - (mean, stdev) tuples for each continuous feature in the dataset
      - {featureValue: probability} for each discrete feature in the dataset
    '''
    summaries = []
    clusters = zip(*dataset)
    del clusters[-1] # delete the cluster of y values as we need not compute anything for these
    for index, cluster in enumerate(clusters):
        if featureTypes[index]:
            # this feature is continuous
            summaries.append((mean(cluster), stdev(cluster)))
        else:
            # this feature is discrete
            summaries.append(getDiscreteProbabilities(cluster))
    return summaries

def summarizeByClass(featureTypes, dataset):
    '''
    returns a map of class value to:
    {
        featureStatistics: [(mean, variance) per attribute for the class],
        probability: class probability
    }
    '''
    classProbabilities = getClassProbabilities(dataset)
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = {'featureStatistics': summarize(featureTypes, instances), 'probability': classProbabilities[classValue]}
    return summaries

##
# We can use a guassian function to estimate the probability of a given feature
# value, given its mean and standard deviation as estimated from the training data.
# Because the mean and standard deviations for each feature were calculated
# separately for each class, the estimated probability of the feature is the
# conditional probability of the feature given the class, which is precisely what
# we need.
##

def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2)/(2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def calculateProbabilitiesByClass(summaries, inputVector):
    probabilities = {}
    for classValue, classSummary in summaries.iteritems():
        probabilities[classValue] = classSummary['probability']
        for i in range(len(classSummary['featureStatistics'])):
            currProb = 0.0
            currFeatureValue = inputVector[i]
            if type(classSummary['featureStatistics'][i]) == types.TupleType:
                mean, stdev = classSummary['featureStatistics'][i]
                currProb = calculateProbability(currFeatureValue, mean, stdev)
            else:
                if currFeatureValue in classSummary['featureStatistics'][i]:
                    currProb = classSummary['featureStatistics'][i][currFeatureValue]
                else:
                    print("WARNING: Bayesian probability that input vector {0} has class " +
                        "{1} is zero as the training data contained no instances with " +
                        "feature {2} given class {1}").format(inputVector, classValue, i)
            probabilities[classValue] *= currProb
    return probabilities

##
# Now, in order to classify an instance, we merely need to iterate through
# the probabilities dictionary and select the class with greatest probability
##

def predict(summaries, inputVector):
    '''
    given the class summaries and an input vector, predicts the class of the input
    '''
    probabilities = calculateProbabilitiesByClass(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if probability > bestProb:
            bestLabel = classValue
            bestProb = probability
    return bestLabel

def getPredictions(summaries, testSet):
    '''
    returns an array of classification predictions for the instances in a test set
    '''
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

##
# We'll want to know how accurate our predictions are
##

def getAccuracy(testSet, predictions):
    '''
    returns the percentage of the test set that were classified correctly
    '''
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0

def runNaiveBayes():
	#filename = 'pima-indians.data.csv'
	#splitRatio = 0.67
	#dataset = loadCsv(filename)
    trainingSet = [[1,20,0], [0,12,1], [1,25,0], [0,10,1], [1, 18, 1], [1, 22, 0], [0, 14, 0]]
    testSet = [[0, 14, 1], [1, 21, 0]]
    featureTypes = [False, True]
	#trainingSet, testSet = splitDataset(dataset, splitRatio)
	#print('Split {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
    summaries = summarizeByClass(featureTypes, trainingSet)
    print('Summaries: {0}').format(summaries)
    predictions = getPredictions(summaries, testSet)
    print('Predictions: {0}').format(predictions)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy: {0}%').format(accuracy)

runNaiveBayes()
