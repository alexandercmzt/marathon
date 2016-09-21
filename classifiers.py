# This file was written by Alexander Chatron-Michaud, student ID 260611509
# COMP 551 Fall 2016
# Assignment 1
# Usage of classes is described below in a comment at the start of each class

import numpy as np

class LinearRegressor:
	#USAGE: Declare the class with initial parameters X, y, and optionally alpha (necessary for hyperparameter tuning)
	#		Train model by calling train, i.e. if your regressor is called r1 then do r1.train() or r1.train(gd=False) if you want to use closed form solution
	#		Perform prediction on a test matrix X by calling r1.predict(X)

	def __init__(self,X,y,alpha=0.003):
		self.y = y
		self.alpha = alpha
		self.m = X.shape[0]
		try:
			self.n = X.shape[1]
		except IndexError:
			self.n = 1

		#Add bias term to features in X
		new_X = np.ones((self.m,self.n+1))
		new_X[:,1:] = X
		self.X = new_X
		self.w = np.random.rand(self.n+1)

	def predict(self,X):
		try:
			retvar = X.dot(self.w)
		except ValueError:
			new_X = np.ones((X.shape[0],self.n+1))
			new_X[:,1:] = X
			X = new_X
			retvar = X.dot(self.w)
		return retvar

	def cost(self,X, y, w = self.w):
		prediction_matrix = self.predict(X)
		difference = y - prediction_matrix
		return difference.T.dot(difference)

	def deriv_cost(self,X, y, w):
		return 2*(X.T.dot(X).dot(w)-X.T.dot(y))

	def closed_form_solve(self,X ,y):
		self.w = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
		return self.w

	def gradient_descent(self,X, y):
		current_best = self.cost(self.X, self.y, self.w)
		cost_history = []
		for _ in xrange(1000):
			self.w = self.w - self.alpha*self.deriv_cost(self.X, self.y, self.w)
			this_cost = self.cost(self.X, self.y, self.w)
			if abs(current_best - this_cost) < 0.001:
				cost_history.append(this_cost)
				return self.w, cost_history
			else:
				cost_history.append(this_cost)
				current_best = this_cost
				if len(cost_history) > 4:
					if cost_history[-1] > cost_history[-2] and cost_history[-2] > cost_history[-3] and cost_history[-3] > cost_history[-4]:
						print "PROBLEM IN GRADIENT DESCENT: COST HAS INCREASED FOR 4 CONSECUTIVE ITERATIONS, ALPHA IS TOO LARGE"
						return self.w, cost_history
		print "PROBLEM IN GRADIENT DESCENT:  NO CONVERGENCE AFTER 1000 ITERATIONS, ALPHA IS TOO SMALL"
		return self.w, cost_history

	def train(self,gd=True):
		if not gd:
			wts = self.closed_form_solve(self.X,self.y)
			print "OPTIMAL WEIGHTS: " + str(wts)
		else:
			wts, cost_history = self.gradient_descent(self.X,self.y)
			print "ENDED AFTER " + str(len(cost_history)) + " GRADIENT ITERATIONS"
			print "OPTIMAL WEIGHTS: " + str(wts)
			return cost_history


class LogisticRegressor():
	#USAGE: Declare the class with initial parameters X, y, and optionally alpha (necessary for hyperparameter tuning)
	#		Train model by calling train, i.e. if your regressor is called r1 then do r1.train()
	#		Perform prediction on a test matrix X by calling r1.predict(X)


	def __init__(self,X,y,alpha=0.4):
		self.y = y
		self.alpha = alpha
		self.m = X.shape[0]
		try:
			self.n = X.shape[1]
		except IndexError:
			self.n = 1

		#Add bias term to features in X
		new_X = np.ones((self.m,self.n+1))
		new_X[:,1:] = X
		self.X = new_X
		self.w = np.random.rand(self.n+1)

	def sigmoid(self,x):
			return 1/(1+np.exp(-x))

	def predict(self,X):
		try:
			retvar = self.sigmoid(X.dot(self.w))
		except ValueError:
			new_X = np.ones((X.shape[0],self.n+1))
			new_X[:,1:] = X
			X = new_X
			retvar = self.sigmoid(X.dot(self.w))
		return retvar

	def cost(self, X, y, w = self.w):
		prediction = self.predict(X)
		return -(y.T.dot(np.log(prediction)) + (1-y).T.dot(np.log(1-prediction)))

	def deriv_cost(self, X, y, w):
		prediction = self.predict(X)
		return X.T.dot(prediction-y)

	def gradient_descent(self,X, y):
		current_best = self.cost(self.X, self.y, self.w)
		cost_history = []
		for _ in xrange(1000):
			self.w = self.w - self.alpha*self.deriv_cost(self.X, self.y, self.w)
			this_cost = self.cost(self.X, self.y, self.w)
			if abs(current_best - this_cost) < 0.001:
				cost_history.append(this_cost)
				return self.w, cost_history
			else:
				cost_history.append(this_cost)
				current_best = this_cost
				if len(cost_history) > 4:
					if cost_history[-1] > cost_history[-2] and cost_history[-2] > cost_history[-3] and cost_history[-3] > cost_history[-4]:
						print "PROBLEM IN GRADIENT DESCENT: COST HAS INCREASED FOR 4 CONSECUTIVE ITERATIONS, ALPHA IS TOO LARGE"
						return self.w, cost_history
		print "PROBLEM IN GRADIENT DESCENT:  NO CONVERGENCE AFTER 1000 ITERATIONS, ALPHA IS TOO SMALL"
		return self.w, cost_history

	def train(self):
		wts, cost_history = self.gradient_descent(self.X,self.y)
		#print "COST_HISTORY: " + str(cost_history)
		print "Ended after " + str(len(cost_history)) + " iterations"
		print "Optimal weights: " + str(wts)
		return cost_history
