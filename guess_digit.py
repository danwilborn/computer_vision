import os, sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
from corners import FAST

# Initiate feature detectors
orb = cv2.ORB_create()
brisk = cv2.BRISK_create()
akaze = cv2.AKAZE_create()

target_file = 'imgs/training/4/6.png'
query_file = 'query.png'
		
target_img = cv2.imread(target_file)
query_img = cv2.imread(query_file)

# resize target to make it easier to get feature descriptors, query is already appropriate size
target_img = cv2.resize(target_img, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

# get keypoints using FAST corner detector
fast_target = FAST(target_img)
fast_query = FAST(query_img)

target_kp = fast_target.kp
query_kp = fast_query.kp

# fast1.show_result()
# fast2.show_result()

print('######### FAST ##########')
print('----- Key Points -----')
print(target_kp)
print(query_kp)
print(' ')

# use BRISK with keypoints from FAST to generate descriptors
target_brisk_kp, target_brisk_desc = brisk.compute(target_img, target_kp)
query_brisk_kp, query_brisk_desc = brisk.compute(query_img, query_kp)

print('######## BRISK #########')
print('----- Key Points -----')
print(target_brisk_kp)
print(query_brisk_kp)
print(' ')

print('----- Descriptors -----')
print(target_brisk_desc)
print(query_brisk_desc)
print(' ')

############## Brute Force Matching ##############
# create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# Match descriptors
# matches = bf.match(BRISKdesc1, BRISKdesc2)

# Sort matches in order of their distance
# matches = sorted(matches, key = lambda x:x.distance)
# print(matches)

# Draw first 3 matches
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches[:3],None,flags=2)

# plt.imshow(img3),plt.show()

############## KNN Matching ##############
# create BFMatcher object
# bf = cv2.BFMatcher()

# Match descriptors
# matches = bf.knnMatch(BRISKdesc1, BRISKdesc2, k=2)

# Apply ratio test
# good = []
# for m,n in matches:
	# print('m',m.distance)
	# print('n',n.distance)
	# if m.distance < n.distance * 0.8:
		# good.append(m)
# print(good)

# Draw matches
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

# plt.imshow(img3),plt.show()

############## FLANN Matching ##############
FLANN_INDEX_LSH = 6
index_params = dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(target_brisk_desc, query_brisk_desc, k=2)
print(matches)

good = []
#print("finding good matches:")
for match in matches:
	if len(match) == 2:
		#print(match)
		good.append(match)
	#print(match, len(match))
#print("matches after cleaning:")
#print(good)
	
matchesMask = [[0,0] for i in range(len(good))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(good):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

'''
if len(good) > MIN_MATCH_COUNT:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()

	h, w = img1.shape
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	dst = cv2.perspectiveTransform(pts,M)

	img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3,cv2.LINE_AA)

else:
	print "Not enough matches are found - %d/%d % (len(good),MIN_MATCH_COUNT)
	matchesMask = None

'''

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)




#img3 = cv2.drawMatchesKnn(target_img,target_brisk_kp,query_img,query_brisk_kp,good,None,**draw_params)

plt.imshow(img3),plt.show()
