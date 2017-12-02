import os, sys
import numpy as np
import cv2
from matplotlib import pyplot as plt

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
sift = cv2.SIFT()

features = dict()

for num in img_dict:
	for img in img_dict[num]:
		img = cv2.imread(image, 0)
		kp, desc = sift.detectAndCompute(img, None)
		print(type(desc))
		if num not in features:
			features[num] = list()
		features[num].append(desc)
	