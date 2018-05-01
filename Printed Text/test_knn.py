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
from skimage.feature import hog

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

start_time = time.time()
extract = Extract_Letters()
training_files = ['./ocr/training/training1.png', './ocr/training/training2.png','./ocr/training/training3.png','./ocr/training/training4.png','./ocr/training/training5.png','./ocr/training/training6.png']

folder_string = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz123456789'
name_counter = 600
for files in training_files:
    letters = extract.extractFile(files)
    string_counter = 0
    
    
    for i in letters:
        if string_counter > 60:
            string_counter = 0
        imsave('./training_type_2/' + str(folder_string[string_counter]) + '/' + str(name_counter) + '_snippet.png', i)
        print 'training character: ' + str(folder_string[string_counter]) + ' (' + str(name_counter) + '/' + str(len(letters)) + ')'
        string_counter += 1
        name_counter += 1
print time.time() - start_time, "seconds" 

class OCR_Demo:
    #init of our simple class - self.VARIABLE means that the VARIABLE object is an attribute of the class
    #and we can use it later on. 
    def __init__(self):

        #load the detector
        clf = pickle.load(open("knn.detector","rb"))

        #now load a test image and get the hog features. 
        path_adobe = './ocr/testing/adobe.png'

        extract = Extract_Letters()
        
        letters = extract.extractFile(path_adobe)

        text = "inaregulatorydocumentfiledwiththesectodayannouncedthatchieftechnologyofficerkevinlynchwouldbetakinghisleaveasofthiscomingfridayonmarch182013kevinlynchresignedfromhispositionasexecutivevicepresidentchieftechnologyofficerofadobesystemsincorporatedeffectivemarch222013topursueotheropportunitiesthefilingreadslynchwhocametothecompanyin2005duringitsacquisitionofmacromedialedadobeschargeintosomeofthemorecuttingedgeareasoftechnologyincludingmultiscreencomputingcloudcomputingandsocialmediaforagesadobehadbeenrootedintheworkflowsoftheprintdesigncommunitylynchwasresponsibleforthecompanysshiftintowebpublishingstartingwithdreamweaverhealsooversawadobesresearchandexperiencedesignteamsandwasasadobeputsitinchargeofshapingadobeslongtermtechnologyvisionandfocusinginnovationacrossthecompanyduringatransformativetimerumorsaroundthewebhavepinpointedappleaslynchsnextdestinationanditsnotanentirelynonsensicalrumoradobestransitiontowebtechnologieshasbeennothingifnotprofitableapplestillagiantinconsumerhardwarecoulduseahelpinghandwhenitcomestomultiscreenfluiditysocialmediaandwebbasedsoftware"
   
        result = list()
        data = list()
        answer_string = ""
        for i in letters:
            #read the images
            #image = imread(i,1)
            #flatten it
            #image = imresize(i, (20,20))
            #hog_features = hog(i, orientations=12, pixels_per_cell=(2, 2),cells_per_block=(2, 2))
            #data = clf.predict(hog_features)
            feature_vector =  i.ravel()
            data = clf.predict(feature_vector)
            result.append(data)
            for x in data:
                answer_string += x
        print "THE REAL TEXT: ", text
        print "MY ANSWER TEXT: ", answer_string
            
        score = 0
        for i in xrange(len(text)):
            if text[i] == answer_string[i]:
                score += 1
        print "Score: ", score, "/ ", len(answer_string)
        print "Similarity by index", float(score)/len(text) * 100, "%"

        distance = ratio(answer_string, text) * 100
        print "Similarity by ratio", distance, "%"

        print '\n' 'Results from the knn classifier'
        print '====================================='
        for a in data:
            print 'Probability that query image is ' + str(a)
        print '\n'

def main():
    print __doc__
    OCR_Demo()
    
if __name__ == '__main__':
    main()