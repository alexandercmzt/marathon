import numpy as np
from cross_validation import crossValidate
from classifiers import LinearRegressor as LR
import pickle

# define data (need not be np arrays)
# load X and y from pickle files
def loadPickle(filename):
    print("Loading {0}...").format(filename)
    p = pickle.load(filename)
    print "Done"
    return p

X_lg = loadPickle('X_participantDataForRaceTimes.p')
X_classify = loadPickle('X_participantDataForRaceParticipation.p')
y_lg = loadPickle('Y_montrealMarathonTime.p')
y_classify = loadPickle('Y_montrealMarathonParticipaton.p')

print y_classify

X = np.array([[0.86, 0.09,-0.85,0.87,-0.44,-0.43,-1.10,0.4,-.96,0.17]]).T
y = np.array([2.49,0.83,-0.25,3.1,0.87,0.02,-0.12,1.81,-0.83,0.43])

# define the number of partitions you want
n = 10

# define the model structures you want to test
structures = [
    {
        'powers': [[1]],
        'products': []
    },
    {
        'powers': [[1,2]],
        'products': []
    },
    {
        'powers': [[1,2,3]],
        'products': []
    }
]

# define which regressor you want to use
regressor = LR

# cross validate to get the best model
crossValidate(X, y, structures, regressor, n)
