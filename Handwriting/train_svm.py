import numpy as np 
import os
import itertools
import operator
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature import hog
from skimage import color, exposure
from scipy.misc import imread,imsave,imresize
import numpy.random as nprnd
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
import matplotlib
import pickle

if __name__ == '__main__':
    data = []
    labels = []

    path_all = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz123456789'

    for i in xrange(len(path_all)):
    	path_individual = './training_type/' + path_all[i] + '/'

    	filenames = ([filename for filename in os.listdir(path_individual)])
    	filenames = [path_individual + filename for filename in filenames]

    	for filename in filenames:
    		image = imread(filename,1)
    		#image = imresize(image, (20,20))
    		hog_features = hog(image, orientations=20, pixels_per_cell=(2, 2),
                    cells_per_block=(2, 2))
    		data.append(hog_features)
    		labels.append(path_all[i])
    	
    	print 'Added: ' + path_all[i] + ' sample to the dataset'

    print 'training the svm'

    clf = LinearSVC(dual=False,verbose=1)

    clf.fit(data,labels)

    pickle.dump( clf, open("svm.detector","wb"))

    print 'the size of the data array is: ' + str(len(data))

    data_shape = np.array(data)
    print "shape of data: " + str(data_shape.shape)

    hog_shape  = np.array(hog_features)
    print "shape of hog_features: " + str(hog_shape.shape)