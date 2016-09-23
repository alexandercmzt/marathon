import csv
import numpy as np
from classifiers import LinearRegressor as LinReg
from classifiers import LogisticRegressor
import pickle
from classifiers import NaiveBayes as NB
import math
import time
from cross_validation import crossValidate
from classifiers import LinearRegressor as LogReg
import sys

def iterateAlpha(X, y, regressor):
    alpha_list = []
    start = 0.0000001
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
X_final = loadPickle('data/final_participantDataFinalFor2016.p')
y_classify = y_classify.tolist()
y_classify = np.array(map(float, y_classify))
from sklearn.preprocessing import normalize
X_classify = normalize(X_classify)
X_linreg = normalize(X_linreg)
X_final = normalize(X_final)

if len(X_linreg) != len(y_linreg):
    print "ERROR: regression datasets are of different lengths"

if len(X_classify) != len(y_classify):
    print "ERROR: regression datasets are of different lengths"

print("Using {0} instances for predicting race times").format(len(y_linreg))
print("Using {0} instances for predicting participation").format(len(y_classify))


# define the number of partitions
n = 10

# define the model structures to test
structures = [
    {
        'powers': [ [1] for x in xrange(X_linreg.shape[1]) ],
        'products': []
    },
    {
        'powers': [ [1,2] for x in xrange(X_linreg.shape[1]) ],
        'products': []
    },
    {
        'powers': [ [1,2,3] for x in xrange(X_linreg.shape[1]) ],
        'products': []
    },
    {
        'powers': [ [1,2,3,4] for x in xrange(X_linreg.shape[1]) ],
        'products': []
    },
    {
        'powers': [ [1,2,3,4,5,6] for x in xrange(X_linreg.shape[1]) ],
        'products': []
    },
    {
        'powers': [ [1,2,3,4,5,6,7] for x in xrange(X_linreg.shape[1]) ],
        'products': []
    }

]

IDS = range(0,8711)

#TODO:
#Square features in X_classify here (NAME THE SQUARED VERSION X_classify2)
#Square features in X_linreg here (NAME THE SQUARED VERSION X_linreg2)
#Square features in X_final here (NAME THE SQUARED VERSION X_final2)
#Train the NB and uncomment it below. Don't touch the last line, I need it in that format!

#GET LOGISTIC REGRESSION PREDICTIONS
r1 = LogReg(X_classify2, y_classify)
r1.train()
LOGREG_FINAL = map(str,np.around(r1.predict(X_final2)).tolist())

#GET NAIVE BAYES PREDICTIONS
#Todo: Idk how you run it, you do it :P
#r3 = NB(X_classify, y_classify,???) <-- What's this param/how do I get it?
#r3.train() 
#NB_FINAL = map(str,np.around(r3.predict(X_final)).tolist())

#GET LINEAR REGRESSION PREDICTIONS
r2 = LinReg(X_linreg2, y_linreg)
r2.train()
LINREG_FINAL = r2.predict(X_final2).tolist()
for i,v in enumerate(LINREG_FINAL):
    LINREG_FINAL[i] = time.strftime('%H:%M:%S', time.gmtime(v))


OUTPUT = np.array([IDS, LOGREG_FINAL, NB_FINAL LINREG_FINAL]).T.tolist()
with open("predictions.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(OUTPUT)

print "Successfully outputted predictions to predictions.csv in current directory"


