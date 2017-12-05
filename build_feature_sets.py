import os, sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from corners import FAST

#build dictionary of all the image files in the training set
img_dict = dict()

for subdir, dirs, files in os.walk('imgs\\training'):
	for i in range(10):
		num_dir = os.path.join(subdir, str(i))
		for sub, dir, imgs in os.walk(num_dir):
			for img in imgs:
				if i not in img_dict:
					img_dict[i] = list()
				file_path = os.path.join(num_dir, img)
				#print(file_path)
				img_dict[i].append(file_path)
#print(img_dict)

# Initiate SIFT detector
orb = cv2.ORB_create()
brisk = cv2.BRISK_create()
akaze = cv2.AKAZE_create()

# nfeatures=20, scoreType=cv2.ORB_FAST_SCORE didn't help

features = dict()

# for num in img_dict:
	# for img in img_dict[num]:
		# img = cv2.imread(image, None)
		# kp, desc = orb.detectAndCompute(img, None)
		# print(type(desc))
		# if num not in features:
			# features[num] = list()
		# features[num].append(desc)

file1 = 'imgs/training/4/6.png'
file2 = 'imgs/training/4/881.png'
		
img1 = cv2.imread(file1)
img2 = cv2.imread(file2)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

fast1 = FAST(file1)
fast2 = FAST(file2)

FASTkp1 = fast1.kp
FASTkp2 = fast2.kp

kp1, desc1 = orb.compute(gray1, FASTkp1)
kp2, desc2 = orb.compute(gray2, FASTkp2)
print('######### FAST ##########')
print('----- Key Points -----')
print(FASTkp1)
print(FASTkp2)
print(' ')

print('######### ORB ##########')
print('----- Key Points -----')
print(kp1)
print(kp2)
print(' ')

print('----- Descriptors -----')
print(desc1)
print(desc2)
print(' ')

BRISKkp1, BRISKdesc1 = brisk.compute(gray1, FASTkp1)
BRISKkp2, BRISKdesc2 = brisk.compute(gray2, FASTkp2)

print('######## BRISK #########')
print('----- Key Points -----')
print(BRISKkp1)
print(BRISKkp2)
print(' ')

print('----- Descriptors -----')
print(BRISKdesc1)
print(BRISKdesc2)
print(' ')

# Can only kaze/akaze keypoints 
AKkp1, AKdesc1 = akaze.compute(gray1, None)
AKkp2, AKdesc2 = akaze.compute(gray2, None)

print('######## AKAZE #########')
print('----- Key Points -----')
print(AKkp1)
print(AKkp2)
print(' ')

print('----- Descriptors -----')
print(AKdesc1)
print(AKdesc2)
print(' ')

# create BFMatcher object
#bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Match descriptors
#matches = bf.match(desc1, desc2)

# Sort matches in order of their distance
#matches = sorted(matches, key = lambda x:x.distance)
#print(matches)

# Draw first 3 matches
#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:1],flags=2)

#plt.imshow(img3),plt.show()