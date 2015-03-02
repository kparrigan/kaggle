# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 21:48:33 2014

@author: kparrigan
"""

import numpy  as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import cross_validation
import utils as ut

trainSet  = np.genfromtxt('Data/train.csv', delimiter=',')
labels  = np.genfromtxt('Data/trainLabels.csv', delimiter=',')
testSet  = np.genfromtxt('Data/test.csv', delimiter=',')

#scale the data before training/testing with SVM
#accuracy decreases with scaling using this method
trainSet = preprocessing.scale(trainSet)
testSet = preprocessing.scale(testSet)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(trainSet, labels, test_size=0.4, random_state=0)

classifier = RandomForestClassifier(n_estimators=100000, max_features=None)
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)
#predictions = classifier.predict(testSet)
#ut.makeSubmissionFile(predictions, 'Submissions/', "Id,Solution")

print ('Done')