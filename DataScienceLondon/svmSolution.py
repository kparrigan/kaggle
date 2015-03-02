# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 19:29:37 2014

@author: kparrigan
"""

import numpy  as np
from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation as cv
from sklearn import decomposition
from sklearn import grid_search as gs
from sklearn import metrics
import utils as ut

def decomposition_pca(train, test):
    """ Linear dimensionality reduction """
    pca = decomposition.PCA(n_components=12, whiten=True)
    train_pca = pca.fit_transform(train)
    test_pca = pca.transform(test)
    return train_pca, test_pca
    
def split_data(X_data, y_data):
    """ Split the dataset in train and test """
    return cv.train_test_split(X_data, y_data, test_size=0.1, random_state=0)

def grid_search(y_data):
    c_range = [1]
    #c_range = 10.0 ** np.arange(1,5,.25) #originally 6.5,7.5,.25
    gamma_range = 10.0 ** np.arange(-1.5,0.5,.25) #originally -1.5,0.5,.25
    params = [{'kernel': ['rbf'], 'gamma': gamma_range, 'C': c_range }] #'class_weight': ['auto']

    cvk = cv.StratifiedKFold(y_data, n_folds=5)
    return gs.GridSearchCV(svm.SVC(), params, cv=cvk)

def train(features, result):
    """ Use features and result to train Support Vector Machine"""
    clf = grid_search(result)
    clf.fit(features, result)
    return clf
  
def predict(clf, features):
    """ Predict labels from trained CLF """
    return clf.predict(features).astype(np.int)
  
def show_score(clf, X_test, y_test):
    """ Scores are computed on the test set """
    y_pred = predict(clf, X_test)
    print metrics.classification_report(y_test.astype(np.int), y_pred)


X_data  = np.genfromtxt('Data/train.csv', delimiter=',')
y_data  = np.genfromtxt('Data/trainLabels.csv', delimiter=',')
test_data  = np.genfromtxt('Data/test.csv', delimiter=',')

trainSet = preprocessing.scale(X_data)
testSet = preprocessing.scale(test_data)

#X_train, X_test, y_train, y_test = cv.train_test_split(trainSet, labels, test_size=0.1, random_state=0)
X_data, test_data = decomposition_pca(X_data, test_data)
X_train, X_test, y_train, y_test = split_data(X_data, y_data)
classifier = train(X_train, y_train)
show_score(classifier, X_test, y_test)

#classifier = svm.SVC(C=10.0,kernel='rbf')
#classifier.fit(trainSet, labels)
#score = classifier.score(X_test, y_test)
#classifier.fit(trainSet, labels)
#predictions = classifier.predict(test_data)
#ut.makeSubmissionFile(predictions, 'Submissions/', "Id,Solution")

print ('Done')