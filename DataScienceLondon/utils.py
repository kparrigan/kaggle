# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 16:19:34 2014

@author: kparrigan
"""
import uuid
import numpy  as np

def makeSubmissionFile(labels, directory, headers):
    guid = uuid.uuid1() 
    fileName = directory + 'submission' + '.csv'
    indices = np.asarray(range(1,len(labels)+1))
    labels = np.column_stack((indices, labels))
    np.savetxt(fileName, labels, delimiter=",", fmt='%i', header=headers, comments='')
