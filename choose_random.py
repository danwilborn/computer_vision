import sys
import os
from random import randint

### Script randomly selects an image from training data set and displays it ###
# directory structure is ./imgs/trainging/{int}/****.png

training_dir = './imgs/training'

# chose what digit to display
num = randint(0, 9)
print 'digit ' + str(num)

# get list of all the pngs in that directory
path = os.path.join(training_dir, str(num))
img_list = [x for x in os.listdir(path) if x.endswith('.png')]

# there are 10 training images per digit so chose another random number between 0 and 9
index = randint(0, 9)
img = img_list[index]
img_path = os.path.join(path, img)
print img_path

# display that image
cmd = 'display {}'.format(img)
os.system(cmd)
