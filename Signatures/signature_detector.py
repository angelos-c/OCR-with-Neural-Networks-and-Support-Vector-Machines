import numpy as np 
from skimage.feature import hog
from scipy.misc import imread,imresize
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

if __name__ == '__main__':
	 #load the detector
	clf = pickle.load( open("place.detector","rb"))
	 
	path_testing = './testing/'

	testing_filenames_plain = sorted([filename for filename in os.listdir(path_testing) if (filename.endswith('.jpg')) ])

	testing_filenames = [path_testing+filename for filename in testing_filenames_plain]
	i = 0
	answers = []
	predictions = []

	for filename in testing_filenames:
		image = imread(filename,1)
		image = imresize(image, (200,200))

		hog_features = hog(image, orientations=12, pixels_per_cell=(16, 16),
					cells_per_block=(1, 1))
		result = clf.predict(hog_features)

		print 'Based on filename, image ' + str(i) + ' is actually -> ' + str(testing_filenames_plain[i])
		
		name = str(testing_filenames_plain[i])
		if "angelos" in name:
			answers.append("angelos")
		elif "tim" in name:
			answers.append("tim")
		elif "hank" in name:
			answers.append("hank")

		i = i + 1

		if result == 0:
			print("The SVM predicts this signature is ANGELOS")
			print("======================================")
			predictions.append("angelos")

		elif result == 1:
			print("The SVM predicts this signature is TIM")
			print("======================================")
			predictions.append("tim")

		elif result == 2:
			print("The SVM predicts this signature is HANK")
			print("======================================")
			predictions.append("hank")

		else:
			print("Something went wrong")
print 'Finished printing testing data'

print "Correct answers: " , answers
print "SVM Predictions: ", predictions

score = 0

for i in range(len(answers)):
	if answers[i] == predictions[i]:
		score = score +1

percentage = score/len(answers) * 100
print 'Signature recognition accuracy: ' + str(percentage) + '%'


