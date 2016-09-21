# This module implements cross validation of prospective data models

import numpy as np
import random
from classifiers import LinearRegressor as LinReg
from classifiers import LogisticRegressor as LogReg
from classifiers import NaiveBayes as NB

class FeatureMatrixGenerator(object):
    '''
    This class generates feature matrices given a dataset an a datastructure
    indicating the powers and product terms you want in the output matrix
    '''
    def __init__(self, X, structure):
        ''' X is a 2d array '''
        self.X = X
        self.structure = structure

    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        if not value:
            raise Exception("X cannot be empty")
        elif len(value) == 0:
            raise Exception("X must contain data")
        elif len(value[0]) == 0:
            raise Exception("X must contain data")
        self._X = value

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        if len(value['powers']) != len(self.X[0]):
            raise Exception("Structure dimension is off")
        self._structure = value

    def deduceFeatureTypes(self, basicFTypes):
        fTypes = []
        # power terms
        powerTerms = self.structure['powers']
        for i, powers in enumerate(powerTerms):
            currType = basicTypes(i)
            for x in range(len(powers)):
                fTypes.append(currType)

        # product terms - these should only ever be products of continuous types
        products = self.structure['products']
        for i in range(len(products)):
            fTypes.append(True)

        return fTypes

    def generate(self):
        '''
        Uses self.structure and to extend self.X
        self.structure has the following form:
          {
            powers: [list at index i contains powers desired for feature i],
            products: [a list of lists of numbers, each list indicating a product of the features in the list]
          }
        '''
        featureMatrix = []
        # generate power terms
        powers = self.structure['powers']
        for currFeature, currPowers in enumerate(powers):
            for currPower in currPowers:
                for i, v in enumerate(self.X):
                    if i >= len(featureMatrix):
                        featureMatrix.append([])
                    featureMatrix[i].append(float(v[currFeature]) ** currPower)

        # generate product terms
        products = self.structure['products']
        for product in products:
            for index, featureVector in enumerate(self.X):
                prod = 1.0
                for feature in product:
                    prod *= featureVector[feature]
                featureMatrix[index].append(prod)

        # convert to numpy array
        retX = np.array(featureMatrix)
        return retX

class DataPartitioner:
    '''
    This class partitions a dataset X, y into n parts of equal size
    partitions, the key datastructure of this class, is a list of two-element
    lists: [X instances, y instances]
    '''

    def __init__(self, n, X, y):
        self.n = n
        self.partitions = self.partition(n, X, y)

    def partition(self, n, X, y):
        chunks = []
        for i in range(len(X)):
            partitionIndex = i % n
            if partitionIndex >= len(chunks):
                chunks.append([[],[]])
            chunks[partitionIndex][0].append(X[i])
            chunks[partitionIndex][1].append(y[i])
        return chunks

    def getPartitions(self, i):
        ''' returns the ith partition as the validation set and the remaining
        partitions as the training set '''
        validationSet = self.partitions[i]
        trainingSet = [[],[]]
        for index, partition in enumerate(self.partitions):
            if index != i:
                trainingSet[0].extend(partition[0])
                trainingSet[1].extend(partition[1])
        return trainingSet, validationSet

####################
# Cross validation
####################

def get_avg_errors(errs):
    ''' returns average training and validation errors given an array of pairs '''
    totals = [0.0,0.0]
    for i,v in enumerate(errs):
        totals[0] += v[0]
        totals[1] += v[1]
    return map(lambda x: x/len(errs), totals)

def get_index_of_min_err(data):
    ''' takes array of [avg training error, avg validation error] entries, and returns index with the lowest validation error '''
    v_errs = map(lambda (x,y): y, data)
    min_err = min(v_errs)
    return v_errs.index(min_err)

def crossValidate(X, y, modelStructures, regressor, n):
    ''' takes in data and a list of model structures and returns the best model
    X should be a matrix with one feature vector per row
    y should be a list of the outputs corresponding to each feature vector in X
    modelStructures is a list of dictionaries, each of which describes a different feature set to be tested
    regressor is a reference to whichever regression class you want to use (e.g. regressor = LinReg)
    n is the number of partitions desired
    '''
    dataPartitioner = DataPartitioner(n, X, y)
    errors = []
    for structure in modelStructures:
        currModelErrors = []
        for i in range(n):
            trainingSet, validationSet = dataPartitioner.getPartitions(i)
            fmg = FeatureMatrixGenerator(trainingSet[0], structure)
            X = fmg.generate()
            y = np.array(trainingSet[1])
            lr = regressor(X, y)
            lr.train()
            trainErr = lr.cost(X, y)
            fmgVal = FeatureMatrixGenerator(validationSet[0], structure)
            XVal = fmgVal.generate()
            yVal = np.array(validationSet[1])
            validationErr = lr.cost(XVal, yVal)
            currModelErrors.append([trainErr, validationErr])
        errors.append(get_avg_errors(currModelErrors))

    bestModelIdx = get_index_of_min_err(errors)
    print("Best model index: {0} \nBest model structure: {1}").format(bestModelIdx, modelStructures[bestModelIdx])
    print "Model errors:\nmodel\ttrain\tvalidation"
    for i, v in enumerate(errors):
        print("{0}\t{1}\t{2}").format(i, v[0], v[1])
    return bestModelIdx

def crossValidateNB(X, y, featureTypes, modelStructures, n):
    dataPartitioner = DataPartitioner(n, X, y)
    errors = []
    for structure in modelStructures:
        currModelErrors = []
        for i in range(n):
            trainingSet, validationSet = dataPartitioner.getPartitions(i)
            fmg = FeatureMatrixGenerator(trainingSet[0], structure)
            X = fmg.generate()
            y = np.array(trainingSet[1])
            fTypes = fmg.deduceFeatureTypes(featureTypes)
            nb = NB(X, y, fTypes)
            nb.train()
            tPredictions = nb.predict(X, y)
            tErr = nb.error(X, y, tPredictions)
            fmgVal = FeatureMatrixGenerator(validationSet[0], structure)
            XVal = fmgVal.generate()
            yVal = np.array(validationSet[1])
            vPredictions = nb.predict(XVal, yVal)
            vErr = nb.error(XVal, yVal, vPredictions)
            currModelErrors.append([tErr, vErr])
        errors.append(get_avg_errors(currModelErrors))

    bestModelIdx = get_index_of_min_err(errors)
    print("Best model index: {0} \nBest model structure: {1}").format(bestModelIdx, modelStructures[bestModelIdx])
    print "Model errors:\nmodel\ttrain\tvalidation"
    for i, v in enumerate(errors):
        print("{0}\t{1}\t{2}").format(i, v[0], v[1])
    return bestModelIdx
