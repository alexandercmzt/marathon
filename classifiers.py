# This file was written by Alexander Chatron-Michaud, student ID 260611509
# COMP 551 Fall 2016
# Assignment 1
# Usage of classes is described below in a comment at the start of each class

import numpy as np
from numpy import e
import math
import types
from scipy.special import expit

class LinearRegressor:
	#USAGE: Declare the class with initial parameters X, y, and optionally alpha (necessary for hyperparameter tuning)
	#		Train model by calling train, i.e. if your regressor is called r1 then do r1.train() or r1.train(gd=False) if you want to use closed form solution
	#		Perform prediction on a test matrix X by calling r1.predict(X)

    def __init__(self,X,y,alpha=0.003):
    	self.y = y
    	self.alpha = alpha
    	self.m = X.shape[0]
    	try:
    		self.n = X.shape[1]
    	except IndexError:
    		self.n = 1

    	#Add bias term to features in X
    	new_X = np.ones((self.m,self.n+1))
    	new_X[:,1:] = X
    	self.X = new_X
    	self.w = np.random.rand(self.n+1)

    def predict(self,X):
    	try:
    		retvar = X.dot(self.w)
    	except ValueError:
    		new_X = np.ones((X.shape[0],self.n+1))
    		new_X[:,1:] = X
    		X = new_X
    		retvar = X.dot(self.w)
    	return retvar

    def cost(self,X, y):
        w = self.w
    	prediction_matrix = self.predict(X)
    	difference = y - prediction_matrix
    	return (1/float(X.shape[0]))*difference.T.dot(difference)

    def deriv_cost(self,X, y, w):
    	return (1/float(X.shape[0]))*2*(X.T.dot(X).dot(w)-X.T.dot(y))

    def closed_form_solve(self,X ,y):
    	self.w = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    	return self.w

    def gradient_descent(self,X, y):
    	current_best = self.cost(self.X, self.y)
    	cost_history = []
    	for _ in xrange(10000):
    		self.w = self.w - self.alpha*self.deriv_cost(self.X, self.y, self.w)
    		this_cost = self.cost(self.X, self.y)
    		if abs(current_best - this_cost) < 0.0001:
    			cost_history.append(this_cost)
    			return self.w, cost_history
    		else:
    			cost_history.append(this_cost)
    			current_best = this_cost
    			if len(cost_history) > 4:
    				if cost_history[-1] > cost_history[-2] and cost_history[-2] > cost_history[-3] and cost_history[-3] > cost_history[-4]:
    					print "ALPHA TOO LARGE"
    					return self.w, cost_history
    	print "ALPHA TOO SMALL"
    	return self.w, cost_history

    def train(self,gd=False):
    	if not gd:
    		wts = self.closed_form_solve(self.X,self.y)
            	print "COST:" + str(self.cost(self.X,self.y))
    		print "OPTIMAL WEIGHTS: " + str(wts)
    	else:
    		wts, cost_history = self.gradient_descent(self.X,self.y)
    		print "Ended after" + str(len(cost_history)) + "iterations"
	        print "COST", self.cost(self.X,self.y)
    		print "Optimal weights: " + str(wts)
    		return cost_history


class LogisticRegressor():
	#USAGE: Declare the class with initial parameters X, y, and optionally alpha (necessary for hyperparameter tuning)
	#		Train model by calling train, i.e. if your regressor is called r1 then do r1.train()
	#		Perform prediction on a test matrix X by calling r1.predict(X)

    def __init__(self,X,y,alpha=1):
    	self.y = y
    	self.alpha = alpha
    	self.m = X.shape[0]
    	try:
    		self.n = X.shape[1]
    	except IndexError:
    		self.n = 1

    	#Add bias term to features in X
    	new_X = np.ones((self.m,self.n+1))
    	new_X[:,1:] = X
    	self.X = new_X
    	self.w = np.random.rand(self.n+1)

    def sigmoid(self,x):
        return 1.0/(1.0+(np.exp(-x)))

    def predict(self,X):
    	try:
    		retvar = self.sigmoid(X.dot(self.w))
    	except ValueError:
    		new_X = np.ones((X.shape[0],self.n+1))
    		new_X[:,1:] = X
    		X = new_X
    		retvar = self.sigmoid(X.dot(self.w))
    	return retvar

    def cost(self, X, y):
        w = self.w
        prediction = self.predict(X)
        return (1/float(X.shape[0]))*(-y.T.dot(np.log(prediction)) - (1-y).T.dot(np.log(1-prediction)))

    def deriv_cost(self, X, y, w):
        prediction = self.predict(X)
        return X.T.dot(prediction-y)

    def gradient_descent(self,X, y):
    	current_best = self.cost(self.X, self.y)
    	cost_history = []
    	for _ in xrange(10000):
    		self.w = self.w - self.alpha*self.deriv_cost(self.X, self.y, self.w)
    		this_cost = self.cost(self.X, self.y)
    		if abs(current_best - this_cost) < 0.00001:
    			cost_history.append(this_cost)
    			return self.w, cost_history
    		else:
    			cost_history.append(this_cost)
    			current_best = this_cost
    			if len(cost_history) > 4:
    				if cost_history[-1] > cost_history[-2] and cost_history[-2] > cost_history[-3] and cost_history[-3] > cost_history[-4]:
    					print "ALPHA IS TOO LARGE"
    					return self.w, cost_history
    	print "ALPHA IS TOO SMALL"
    	return self.w, cost_history

    def train(self,gd=True):
    	wts, cost_history = self.gradient_descent(self.X,self.y)
    	#print "COST_HISTORY: " + str(cost_history)
    	print "Ended after " + str(len(cost_history)) + " iterations"
    	print "Optimal weights: " + str(wts)
        print self.cost(self.X, self.y)
    	return cost_history


class NaiveBayes:
    ''' Usage:
    call train() to save the probabilities on the class
    call predict(dataSet) to get the predictions for the given vectors
    call getAccuracy(dataSet, predictions) to get the percentage accuracy of the predictions
      dataSet should be the same set passed to predict
      predictions should be the return from predict
    '''

    def __init__(self, X, y, featureTypes):
        '''
        featureTypes is a list indicating true at index i if feature i is continuous, and false
        if feature i is discrete
        trainingSet is a matrix where each entry is a vector with nth entry = y and previous n-1 entries = feature values
        '''
        self.trainingSet = self.compose(X, y)
        self.featureTypes = featureTypes

    def compose(self, X, y):
        composed = []
        for i, v in enumerate(X):
            entry = np.append(v, y[i])
            composed.append(entry)
        return composed

    def separateByClass(self, dataset):
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

    def mean(self, numbers):
    	return sum(numbers)/float(len(numbers))

    def stdev(self, numbers):
    	avg = self.mean(numbers)
    	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    	return math.sqrt(variance)

    def getClassProbabilities(self, dataset):
        '''
        returns a map of class to its probability, as computed from dataset
        '''
        probabilities = {}
        dataSetSize = len(dataset)
        separated = self.separateByClass(dataset)
        for classValue, instances in separated.iteritems():
            probabilities[classValue] = float(len(instances))/dataSetSize
        return probabilities

    def getDiscreteProbabilities(self, numbers):
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

    def summarize(self, featureTypes, dataset):
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
                summaries.append((self.mean(cluster), self.stdev(cluster)))
            else:
                # this feature is discrete
                summaries.append(self.getDiscreteProbabilities(cluster))
        return summaries

    def summarizeByClass(self, featureTypes, dataset):
        '''
        returns a map of class value to:
        {
            featureStatistics: [(mean, variance) per attribute for the class],
            probability: class probability
        }
        '''
        classProbabilities = self.getClassProbabilities(dataset)
        separated = self.separateByClass(dataset)
        summaries = {}
        for classValue, instances in separated.iteritems():
            summaries[classValue] = {'featureStatistics': self.summarize(featureTypes, instances), 'probability': classProbabilities[classValue]}
        return summaries

    ##
    # We can use a guassian function to estimate the probability of a given feature
    # value, given its mean and standard deviation as estimated from the training data.
    # Because the mean and standard deviations for each feature were calculated
    # separately for each class, the estimated probability of the feature is the
    # conditional probability of the feature given the class, which is precisely what
    # we need.
    ##

    def calculateProbability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2)/(2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

    def calculateProbabilitiesByClass(self, summaries, inputVector):
        probabilities = {}
        for classValue, classSummary in summaries.iteritems():
            probabilities[classValue] = classSummary['probability']
            for i in range(len(classSummary['featureStatistics'])):
                currProb = 0.0
                currFeatureValue = inputVector[i]
                if type(classSummary['featureStatistics'][i]) == types.TupleType:
                    mean, stdev = classSummary['featureStatistics'][i]
                    currProb = self.calculateProbability(currFeatureValue, mean, stdev)
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

    def predictSingleInstance(self, summaries, inputVector):
        '''
        given the class summaries and an input vector, predicts the class of the input
        '''
        probabilities = self.calculateProbabilitiesByClass(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.iteritems():
            if probability > bestProb:
                bestLabel = classValue
                bestProb = probability
        return bestLabel

    def predictSingleInstanceDynamic(self, summaries, inputVector, ratio):
        '''
        this method only applies to binary classification
        given the class summaries and an input vector, predicts the class of the input
        ratio allows for a flexible decision boudary as follows:
        a ratio of n indicates that if (prob positive classification/prob negative classification) >= n
        then the instance will be classified as positive (1), and negative (0) otherwise
        '''
        probabilities = self.calculateProbabilitiesByClass(summaries, inputVector)
        positiveProb = probabilities[1]
        negativeProb = probabilities[0]
        if (positiveProb / negativeProb) >= ratio:
            return 1
        return 0

    def getPredictions(self, summaries, testSet):
        '''
        returns an array of classification predictions for the instances in a test set
        '''
        predictions = []
        for i in range(len(testSet)):
            result = self.predictSingleInstance(summaries, testSet[i])
            predictions.append(result)
        return predictions

    def getPredictionsForDynamicDecisionBoundary(self, summaries, testSet, ratio):
        predictions = []
        for i in range(len(testSet)):
            result = self.predictSingleInstanceDynamic(summaries, testSet[i], ratio)
            predictions.append(result)
        return predictions

    ##
    # We'll want to know how accurate our predictions are
    ##

    def getAllStats(self, testSet, predictions):
        '''
        returns a bunch of statistics about the quality of the classification
        '''
        total_entries = len(predictions)
        total_incorrect = 0
        actual_class_0 = 0
        predicted_class_0 = 0
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == 0:
                actual_class_0 += 1
            if predictions[i] == 0:
                predicted_class_0 += 1
            if testSet[i][-1] == predictions[i]:
                if predictions[i] == 0:
                    true_negatives += 1
                else:
                    true_positives += 1
            else:
                total_incorrect += 1
                if testSet[i][-1] == 0:
                    false_positives += 1
                else:
                    false_negatives += 1

        # return % error, total entries, total correct, total incorrect, total actual class 0, total predicted class 0, same for class 1, true positives, false positives, true negatives, false negatives
        return [float(total_incorrect)/total_entries * 100.0, total_entries, total_entries - total_incorrect, total_incorrect, actual_class_0, predicted_class_0, len(testSet) - actual_class_0, len(testSet) - predicted_class_0, true_positives, false_positives, true_negatives, false_negatives]

    def getAccuracy(self, testSet, predictions):
        '''
        returns the percentage of the test set that were classified correctly
        '''
        total_incorrect = 0
        for i in range(len(testSet)):
            if testSet[i][-1] != predictions[i]:
                total_incorrect += 1

        return (total_incorrect/float(len(testSet))) * 100.0

    def train(self):
        self.summaries = self.summarizeByClass(self.featureTypes, self.trainingSet)
        return self.summaries

    def predict(self, X, y):
        dataSet = self.compose(X, y)
        return self.getPredictions(self.summaries, dataSet)

    def predictDynamic(self, X, y, ratio):
        dataSet = self.compose(X, y)
        return self.getPredictionsForDynamicDecisionBoundary(self.summaries, dataSet, ratio)

    def cost(self, X, y, predictions):
        testSet = self.compose(X, y)
        return self.getAccuracy(testSet, predictions)

    def getExtendedStats(self, X, y, predictions):
        testSet = self.compose(X, y)
        return self.getAllStats(testSet, predictions)
