import numpy as np
from classifiers import LinearRegressor as LR
from classifiers import LogisticRegressor as LOGS
X = np.array([[0.86, 0.09,-0.85,0.87,-0.44,-0.43,-1.10,0.4,-.96,0.17]]).T
y = np.array([2.49,0.83,-0.25,3.1,0.87,0.02,-0.12,1.81,-0.83,0.43])
X1 = np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]).T
y1 = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
a = LR(X,y)
a.train()
a.predict(X)
b = LOGS(X1,y1)
b.train()
b.predict(X1)