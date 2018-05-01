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
from sklearn.neighbors import KNeighborsClassifier


if __name__ == '__main__':

    data = []
    labels = []

    text = "abcdefghijklmnopqrstuvwxyz123456789"

    for i in xrange(len(text)):
        path = './training_type/' + text[i] + '/'
        
        filenames = ([filename for filename in os.listdir(path)])
        

        filenames = [path+filename for filename in filenames]
    
        for filename in filenames:
            image = imread(filename,1)
            #hog_features = hog(image, orientations=12, pixels_per_cell=(2, 2),
            #    cells_per_block=(2, 2))
            data.append(image.ravel())
            labels.append(text[i])
        print 'Finished adding ' + text[i] + ' samples to dataset'

    print 'Training the KNN classifier'
    #create the KNC
    clf = KNeighborsClassifier(n_neighbors=5)
    #train the KNN
    clf.fit(data, labels)
    print 'Trained the KNN-classifier'

    #size = hog_features.shape
    print 'size of data array is: ' + str(len(data))


    #pickle it - save it to a file
    pickle.dump( clf, open( "knn.detector", "wb" ) )