import numpy as np
from classifiers import LinearRegressor
from classifiers import LogisticRegressor

def iterateAlpha(X, y):
	alpha_list = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
	for alpha in alpha_list:
		print "[Alpha = " + str(alpha) + "][Linear Regression]"
		r1 = LinearRegressor(X, y, alpha)
		r1.train()
	for alpha in alpha_list:
		print "[Alpha = " + str(alpha) + "][Logistic Regression]"
		r1 = LogisticRegressor(X, y, alpha)
		r1.train()

