import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.misc import imread,imresize,imsave
from skimage.segmentation import clear_border
from skimage.morphology import label
from skimage.measure import regionprops

class Extract_Letters:
    def extractFile(self, filename):
        image = imread(filename,1)
    
        #apply threshold in order to make the image binary
        bw = image < 120
    
        # remove artifacts connected to image border
        cleared = bw.copy()
        #clear_border(cleared)

        # label image regions
        label_image = label(cleared,neighbors=8)
        borders = np.logical_xor(bw, cleared)
        label_image[borders] = -1
    
    
        fig = plt.figure()
        #ax = fig.add_subplot(131)
        #ax.imshow(bw, cmap='jet')

        letters = list()
        order = list()
    
        for region in regionprops(label_image):
            minc, minr, maxc, maxr = region.bbox
            # skip small images
            if maxc - minc > len(image)/250: # better to use height rather than area.
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
                order.append(region.bbox)


        #sort the detected characters left->right, top->bottom
        lines = list()
        first_in_line = ''
        counter = 0

        #worst case scenario there can be 1 character per line
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
training_files = ['./hand/a.jpg','./hand/b.jpg','./hand/c.jpg','./hand/d.jpg','./hand/e.jpg','./hand/f.jpg','./hand/g.jpg','./hand/h.jpg','./hand/i.jpg','./hand/j.jpg','./hand/k.jpg','./hand/l.jpg','./hand/m.jpg','./hand/n.jpg','./hand/o.jpg','./hand/p.jpg','./hand/q.jpg','./hand/r.jpg','./hand/s.jpg','./hand/t.jpg','./hand/u.jpg','./hand/v.jpg','./hand/w.jpg','./hand/x.jpg','./hand/y.jpg','./hand/z.jpg','./hand/1.jpg','./hand/2.jpg','./hand/3.jpg','./hand/4.jpg','./hand/5.jpg','./hand/6.jpg','./hand/7.jpg','./hand/8.jpg','./hand/9.jpg']

training_file = ['./hand/a.jpg']
folder_string = 'abcdefghijklmnopqrstuvwxyz123456789'
name_counter = 600
counter_big = 0
for files in training_files:
	letters = extract.extractFile(files)
	counter = 0
        for j in letters:
    #imsave('./training_type/a/a' + str(counter) + '_snippet.png',j)
            imsave('./training_type/' + str(folder_string[counter_big]) + '/' + str(folder_string[counter_big]) + str(counter) + '_snippet.png',j)
            counter = counter + 1
            print "done", str(folder_string[counter_big]) + '/' + str(folder_string[counter_big]) + str(counter)
        counter_big = counter_big + 1
print "done"