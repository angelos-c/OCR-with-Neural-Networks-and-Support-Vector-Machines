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
    #paths for the training samples 
    path_angelos = './training/angelos/'
    path_tim = './training/tim/'
    path_hank = './training/hank/'

    angelos_filenames = sorted([filename for filename in os.listdir(path_angelos) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
    tim_filenames = sorted([filename for filename in os.listdir(path_tim) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
    hank_filenames = sorted([filename for filename in os.listdir(path_hank) if (filename.endswith('.jpg') or filename.endswith('.png') or (filename.endswith('.bmp'))) ])
 
    #add the full path to all the filenames
    angelos_filenames = [path_angelos+filename for filename in angelos_filenames]
    tim_filenames = [path_tim+filename for filename in tim_filenames]
    hank_filenames = [path_hank+filename for filename in hank_filenames]

    print 'Number of training images -> angelos: ' + str(len(angelos_filenames))
    print 'Number of training images -> tim: ' + str(len(tim_filenames))
    print 'Number of training images -> hank: ' + str(len(hank_filenames))

    total = len(angelos_filenames) + len(tim_filenames) + len(hank_filenames)

    print 'Total Number of samples: ' + str(total)

    #create the list that will hold ALL the data and the labels
    #the labels are needed for the classification task:
    # 0 = angelos
    # 1 = tim
    # 2 = hank    
    data = []
    labels = []


    for filename in angelos_filenames:
        #read the images
        image = imread(filename,1)
        #flatten it
        image = imresize(image, (200,200))
        hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        data.append(hog_features)
        labels.append(0)
    print 'Finished adding angelos samples to dataset'
    
    for filename in tim_filenames:
        image = imread(filename,1)
        image = imresize(image, (200,200))
        hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        data.append(hog_features)
        labels.append(1)
    print 'Finished adding tim samples to dataset'

    for filename in hank_filenames:
        image = imread(filename,1)
        image = imresize(image, (200,200))
        hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1))
        data.append(hog_features)
        labels.append(2)
    print 'Finished adding hank samples to dataset'
    
    print 'Training the SVM'
    
    #create the SVC
    clf = LinearSVC(dual=False,verbose=1)
    #train the svm
    clf.fit(data, labels)

    #pickle it - save it to a file
    pickle.dump( clf, open( "place.detector", "wb" ) )


    data_shape = np.array(data)
    print "shape of data: " + str(data_shape.shape)

    hog_shape  = np.array(hog_features)
    print "shape of hog_features: " + str(hog_shape.shape)