# -*- coding: utf-8 -*-
import time
import numpy as np 
from skimage.feature import hog
from scipy.misc import imread,imresize,imsave
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import pickle
import os
import warnings
from skimage.morphology import label
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from Levenshtein import distance,ratio
warnings.filterwarnings("ignore", category=DeprecationWarning) 

class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename,1)
    
        bw = image < 120
    
        cleared = bw.copy()

        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1
    
    
        fig = plt.figure()


        letters = list()
        order = list()
    
        for region in regionprops(label_image):
            minc, minr, maxc, maxr = region.bbox
            # skip small images
            if maxc - minc > len(image)/250: # better to use height rather than area.
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
                order.append(region.bbox)


        lines = list()
        first_in_line = ''
        counter = 0

        for x in range(len(order)):
            lines.append([])
    
        for character in order:
            if first_in_line == '':
                first_in_line = character
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) < (first_in_line[2] - first_in_line[0]):
                lines[counter].append(character)
            elif abs(character[0] - first_in_line[0]) > (first_in_line[2] - first_in_line[0]):
                first_in_line = character
                counter += 1
                lines[counter].append(character)


        for x in range(len(lines)):       
            lines[x].sort(key=lambda tup: tup[1])

        final = list()
        prev_tr = 0
        prev_line_br = 0
        
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                tl_2 = lines[i][j][1]
                bl_2 = lines[i][j][0]
                if tl_2 > prev_tr and bl_2 > prev_line_br:
                    tl,tr,bl,br = lines[i][j]
                    letter_raw = bw[tl:bl,tr:br]
                    letter_norm = imresize(letter_raw ,(20 ,20))
                    final.append(letter_norm)
                    prev_tr = lines[i][j][3]
                if j == (len(lines[i])-1):
                    prev_line_br = lines[i][j][2]
            prev_tr = 0
            tl_2 = 0
        print 'Characters recognized: ' + str(len(final))
        return final


    def __init__(self):
        print "Extracting characters..."


if __name__ == '__main__':
    #load the detector
   clf = pickle.load( open("svm.detector","rb"))

   path_testing = './ocr/testing/adobe.png'

   extract = Extract_Letters()
   letters = extract.extractFile(path_testing)

   text = "inaregulatorydocumentfiledwiththesectodayannouncedthatchieftechnologyofficerkevinlynchwouldbetakinghisleaveasofthiscomingfridayonmarch182013kevinlynchresignedfromhispositionasexecutivevicepresidentchieftechnologyofficerofadobesystemsincorporatedeffectivemarch222013topursueotheropportunitiesthefilingreadslynchwhocametothecompanyin2005duringitsacquisitionofmacromedialedadobeschargeintosomeofthemorecuttingedgeareasoftechnologyincludingmultiscreencomputingcloudcomputingandsocialmediaforagesadobehadbeenrootedintheworkflowsoftheprintdesigncommunitylynchwasresponsibleforthecompanysshiftintowebpublishingstartingwithdreamweaverhealsooversawadobesresearchandexperiencedesignteamsandwasasadobeputsitinchargeofshapingadobeslongtermtechnologyvisionandfocusinginnovationacrossthecompanyduringatransformativetimerumorsaroundthewebhavepinpointedappleaslynchsnextdestinationanditsnotanentirelynonsensicalrumoradobestransitiontowebtechnologieshasbeennothingifnotprofitableapplestillagiantinconsumerhardwarecoulduseahelpinghandwhenitcomestomultiscreenfluiditysocialmediaandwebbasedsoftware"
   answer_string = ""

   result = list()
   data = list()

   for i in letters:
   	hog_features = hog(i, orientations=20, pixels_per_cell=(2, 2),
    	         cells_per_block=(2, 2))
   	data = clf.predict(hog_features)
   	for x in data:
            answer_string += x

   score = 0

   for i in xrange(len(text)):
   	if text[i] == answer_string [i]:
   		score +=1

   print "score : ", score, " / ", len(answer_string)
   print "score by index: ", float(score)/len(text) * 100, "%"

   edit_dist = ratio(answer_string, text) * 100
   print "score by ratio: ", edit_dist, "%"

   print "recognised text: ", answer_string
