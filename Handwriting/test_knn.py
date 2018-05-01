import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops
import pickle
from Levenshtein import distance,ratio

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

class OCR_Demo:
    def __init__(self):

        #load the detector
        clf = pickle.load(open("knn.detector","rb"))

        #now load a test image and get the hog features. 
        path_adobe = './testing/testing.jpg'

        extract = Extract_Letters()
        
        letters = extract.extractFile(path_adobe)

        text = "sometimesthingsgetcomplicated"

        result = list()
        data = list()
        answer_string = ""
        for i in letters:
            #image = imread(i,1)
            #flatten it
            #image = imresize(i, (20,20))
            #hog_features = hog(i, orientations=12, pixels_per_cell=(2, 2),cells_per_block=(2, 2))
            #data = clf.predict_proba(hog_features)
            feature_vector =  i.ravel()
            data = clf.predict(feature_vector)
            #result.append(data)
            for x in data:
                answer_string += x

        print "Actual Original Text: ", text
        print "Recognised Text: ", answer_string
            
        score = 0
        for i in xrange(len(text)):
            if text[i] == answer_string[i]:
                score += 1
        print "Score -> ", score, "/", len(answer_string)
        print "Character index similarity", float(score)/len(text) * 100, "%"

        distance = ratio(answer_string, text) * 100
        print "Similarity by ratio: ", distance, "%"

        print 'ignore below, testing:'
        print '\n' 'Results from the knn classifier'
        print '===================================='
        for a in data:
            print 'Probability that query image is ' + str(a)
        print '\n'

def main():
    print __doc__
    OCR_Demo()
    
if __name__ == '__main__':
    main()