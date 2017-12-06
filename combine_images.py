import os, sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import glob

# get all the training images
# store in img_dict where key is what digit it is and value is list of all the image files
img_dict = dict()
for subdir, dirs, files in os.walk('imgs/training'):
	for i in range(10):
		num_dir = os.path.join(subdir, str(i))
		for sub, dir, imgs in os.walk(num_dir):
			for img in imgs:
				if i not in img_dict:
					img_dict[i] = list()
				file_path = os.path.join(num_dir, img)
				#print(file_path)
				img_dict[i].append(file_path)
			
# generate 50 row images with 100 images per row
# each digit has 500 training images so there are 5 rows of 100 per digit, 50 total
total_rows = 0
row_size = 100
for num in img_dict:
	num_rows = 0
	while num_rows < 5:
		#print('building row {} of number {}'.format(str(num_rows), str(num)))
		# base picture
		base = cv2.imread(img_dict[num][row_size*num_rows])
		himstack = cv2.resize(base, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
		img_count = 1
		while img_count < row_size:
			# calculate index of img file in img_dict[num]
			index = row_size * num_rows + img_count
			img = img_dict[num][index]
			#print('stacking {}'.format(img))
			im = cv2.imread(img)
			im = cv2.resize(im, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
			himstack = np.hstack((himstack, im))
			img_count += 1
		num_rows += 1
		cv2.imwrite("row-" + str(num) + "-" + str(num_rows) + ".png", himstack)
		total_rows += 1
		
# combine all 50 row-#.png files into one image, stacked horizontaly
# use glob to get list of all row image files
os.chdir('.')
row_imgs = [i for i in glob.glob('row*.png')]
# base picture
base = cv2.imread(row_imgs[0])
imstack = base
# sort file names so that rows are in correct order
row_imgs = sorted(row_imgs)
for i,img in enumerate(row_imgs):
	#print(img)
	if i == 0: continue
	im = cv2.imread(img)
	imstack = np.vstack((imstack, im))
# write final query.png
cv2.imwrite("query.png", imstack)

#delete row image files
# bash command:
#cmd = 'rm row*'
# windows command:
cmd = 'del row*'
os.system(cmd)
