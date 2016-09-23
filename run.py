import csv
import numpy as np
from classifiers import LinearRegressor as LinReg
from classifiers import LogisticRegressor
import pickle
from classifiers import NaiveBayes as NB
import math
import time
from cross_validation import crossValidate, FeatureMatrixGenerator
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

# Naive Bayes set up
p1_model = {
    'powers': [ [1] for x in xrange(X_linreg.shape[1]) ],
    'products': []
}

X_classify_nb = X_classify
y_classify_nb = np.array(y_classify)
fmg_nb = FeatureMatrixGenerator(X_classify_nb, p1_model)
X_classify_nb = fmg_nb.generate()

X_final_nb = X_final
fmg_nb_final = FeatureMatrixGenerator(X_final_nb, p1_model)
X_final_nb = fmg_nb_final.generate()

# Lin reg and log reg set up
p2_model = {
    'powers': [ [1,2] for x in xrange(X_linreg.shape[1]) ],
    'products': []
}

fmg_lin_reg = FeatureMatrixGenerator(X_linreg, p2_model)
fmg_classify = FeatureMatrixGenerator(X_classify, p2_model)
fmg_final = FeatureMatrixGenerator(X_final, p2_model)

X_linreg2 = fmg_lin_reg.generate()
X_classify2 = fmg_classify.generate()
X_final2 = fmg_final.generate()

#GET LOGISTIC REGRESSION PREDICTIONS
r1 = LogReg(X_classify2, y_classify)
r1.train()
LOGREG_FINAL = map(str,np.around(r1.predict(X_final2)).tolist())

#GET LINEAR REGRESSION PREDICTIONS
r2 = LinReg(X_linreg2, y_linreg)
r2.train()
LINREG_FINAL = r2.predict(X_final2).tolist()
for i,v in enumerate(LINREG_FINAL):
    LINREG_FINAL[i] = time.strftime('%H:%M:%S', time.gmtime(v))

#GET NAIVE BAYES PREDICTIONS
types = [True, True, True, True, False, True, True, True]
r3 = NB(X_classify_nb, y_classify_nb, types)
r3.train()
y_final_nb = np.array([0 for i in range(len(X_final_nb))])
NB_FINAL = map(str,np.around(r3.predict(X_final_nb, y_final_nb)).tolist())

OUTPUT = np.array([IDS, LOGREG_FINAL, NB_FINAL, LINREG_FINAL]).T.tolist()
with open("predictions.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(OUTPUT)

print "Successfully outputted predictions to predictions.csv in current directory"
