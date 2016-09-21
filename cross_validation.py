# This module implements cross validation of prospective data models

import numpy as np
from random import shuffle
from classifiers import LinearRegressor as LinReg
from classifiers import LogisticRegressor as LogReg

class FeatureMatrixGenerator(object):
    '''
    This class generates feature matrices given a dataset an a datastructure
    indicating the powers and product terms you want in the output matrix
    '''
    def __init__(self, X, y, structure):
        ''' X is a 2d array '''
        self.X = X
        self.y = y
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
        y = np.array(self.y)
        return retX, y

class DataPartitioner:
    '''
    This class partitions a dataset X, ys into n parts of equal size
    '''

    def __init__(self, n, X, ys):
        self.n = n
        self.partitions = self.partition(n, X, ys)

    def partition(self, n, X, ys):
        chunks = self.initializeChunks(n, X, ys)
        for index, instance in enumerate(X):
            partitionIndex = index % n
            chunks[partitionIndex]['X'].append(instance)
            for i, v in enumerate(ys):
                key = "y" + str(i)
                chunks[partitionIndex][key].append(v[index])
        return chunks

    def initializeChunks(self, n, X, ys):
        chunks = []
        for i in range(n):
            chunks.append({'X':[]})
            for j in range(len(ys)):
                key = "y" + str(j)
                chunks[i][key] = []
        return chunks

    def getPartitions(self, i):
        ''' returns the ith partition as the validation set and the remaining
        partitions as the training set '''
        validationSet = self.partitions[i]
        trainingSet = {}
        for index, partition in enumerate(self.partitions):
            if index != i:
                for key, value in partition.iteritems():
                    if key not in trainingSet:
                        trainingSet[key] = []
                    trainingSet[key].extend(value)
        return trainingSet, validationSet

    def getPartitionsForOneY(self, i, y):
        ''' returns the training and validation sets in a convenient format if
        you only want to examine a single y '''
        training, validation = self.getPartitions(i)
        reducedTraining = {}
        reducedValidation = {}
        for key in training.keys():
            if key != 'X' or key != ('y' + str(y)):
                del training[key]
                del validation[key]
        yt = training['y' + str(y)]
        yv = validation['y' + str(y)]
        del training['y' + str(y)]
        del validation['y' + str(y)]
        training['y'] = yt
        validation['y'] = yv
        return training, validation


def crossValidateLinReg(X, ys, modelStructures):
    ''' takes in data and a list of model structures and returns the best model '''
    # partition the data
    n = 10 # number of partitions
    dataPartitioner = DataPartitioner(n, X, ys)
    errors = []
    for structure in modelStructures:
        currModelErrors = []
        for i in range(n):
            trainingSet, validationSet = getPartitionsForOneY(i, 0)
            fmg = FeatureMatrixGenerator(trainingSet['X'], trainingSet['y'])
            X, y = fmg.generate()
            lr = LinReg(X, y)
            lr.train()
            trainErr = lr.
            currModelErrors.append([])



def crossValidateLogReg(X, ys, modelStructures):


    - Loop: for i = 0 to n-1
      - extract the partition i as a test set and the rest of the partitions as a training set using the method described above
      - convert them into a matrix X which corresponds to the model and vector y (y will be the same, but just convert to array with np)
        - **note** NO NEED TO ADD COLUMN OF 1s to X
      - create a LinearRegressor using X and y and train the model
      - compute the error of the model against the training data and the error against the test set and record these errors
    - average training and validation errors over all iterations of loop and record them in a map from model to training error


X = [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7],[8,8],[9,9],[10,10]]
y = [1,2,3,4,5,6,7,8,9,10]
y1 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
dp = DataPartitioner(3, X, [y, y1])
partitions = dp.getPartitions(0)
print "training", partitions[0]
print "validation", partitions[1]
