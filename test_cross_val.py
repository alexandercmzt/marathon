import numpy as np
import pickle
from cross_validation import crossValidate
from classifiers import LinearRegressor as LinReg
from classifiers import LogisticRegressor as LogReg
from classifiers import NaiveBayes as NB
from sklearn.preprocessing import normalize

def iterateAlpha(X, y, regressor):
    alpha_list = []
    start = 1
    for i in range(10):
        alpha_list.append(start)
        start *= 10
    print alpha_list
    for alpha in alpha_list:
		print "[Alpha = " + str(alpha) + "]"
		r1 = regressor(X, y, alpha)
		r1.train()

# load X and y from pickle files
def loadPickle(filename):
    print("Loading {0}...").format(filename)
    p = pickle.load(open(filename, 'rb'))
    print "Done"
    return p

def findBestModel(X, y, models, regressor, n):
    crossValidate(X, y, models, regressor, n)

# script

# load data
X_linreg = loadPickle('data/X_participantDataForRaceTimes.p')
X_classify = loadPickle('data/X_participantDataForRaceParticipation.p')
y_linreg = loadPickle('data/Y_montrealMarathonTime.p')
y_classify = loadPickle('data/Y_montrealMarathonParticipaton.p')

if len(X_linreg) != len(y_linreg):
    print "ERROR: regression datasets are of different lengths"

if len(X_classify) != len(y_classify):
    print "ERROR: regression datasets are of different lengths"

print("Using {0} instances for predicting race times").format(len(y_linreg))
print("Using {0} instances for predicting participation").format(len(y_classify))

# determine alpha
np.set_printoptions(threshold=np.inf)
X_classify = normalize(X_classify)
X_linreg = normalize(X_linreg)

#iterateAlpha(X_classify, y_classify, LogReg) # USE ALPHA=10 for log reg, use train(gd=False) for lin reg!!!

# define the number of partitions
n = 10

# define the model structures to test
structures = [
    {
        'powers': [ [1] for x in xrange(X_linreg.shape[1]) ],
        'products': []
    }
]

regressor = LinReg

#findBestModel()
