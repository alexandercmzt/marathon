import numpy as np
import pickle
from cross_validation import crossValidate
from classifiers import LinearRegressor as LinReg
from classifiers import LogisticRegressor as LogReg
from classifiers import NaiveBayes as NB

def iterateAlpha(X, y, regressor):
	alpha_list = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
	for alpha in alpha_list:
		print "[Alpha = " + str(alpha) + "][Linear Regression]"
		r1 = regressor(X, y, alpha)
		r1.train()

# load X and y from pickle files
def loadPickle(filename):
    print("Loading {0}...").format(filename)
    p = pickle.load(open(filename, 'rb'))
    print "Done"
    return p

X_lg = loadPickle('data/X_participantDataForRaceTimes.p')
X_classify = loadPickle('data/X_participantDataForRaceParticipation.p')
y_lg = loadPickle('data/Y_montrealMarathonTime.p')
y_classify = loadPickle('data/Y_montrealMarathonParticipaton.p')

if len(X_lg) != len(y_lg):
    print "ERROR: regression datasets are of different lengths"

if len(X_classify) != len(y_classify):
    print "ERROR: regression datasets are of different lengths"

print("Using {0} instances for predicting race times").format(len(y_lg))
print("Using {0} instances for predicting participation").fomat(len(y_classify))

def findBestModel(X, y, models, regressor, n):
    crossValidate(X, y, models, regressor, n)

iterateAlpha(X_lg, y_lg, LinReg)

# define the number of partitions
n = 10

# define the model structures to test
structures = [
    {
        'powers': [ [1] for x in xrange(X_lg.shape[1]) ],
        'products': []
    }
]

regressor = LinReg

#findBestModel()
