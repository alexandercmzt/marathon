import numpy as np
import pickle
from cross_validation import getBestDecisionBoundary

# load X and y from pickle files
def loadPickle(filename):
    print("Loading {0}...").format(filename)
    p = pickle.load(open(filename, 'rb'))
    print "Done"
    return p

# load data
X = loadPickle('data/X_participantDataForRaceParticipation.p')
y = loadPickle('data/Y_montrealMarathonParticipaton.p')

if len(X) != len(y):
    print "ERROR: regression datasets are of different lengths"

print("Using {0} instances for predicting participation").format(len(y))

# define the feature types
# NOTE: CHANGE THIS IF THE FEATURES ARE CHANGED!!!!!
types = [True, True, True, True, False, True, True, True]

# define the number of partitions
n = 10

# define the model structures to test
model = {
    'powers': [[1] for x in range(len(X[0]))],
    'products': []
}

getBestDecisionBoundary(X, y, types, model, n)
