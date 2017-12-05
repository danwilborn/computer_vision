import sys
import cv2
import numpy as np

# Corner Harris returns an image dst with the corners in it
class CornerHarris(object):
	def __init__(self, img_file):
		self.img = cv2.imread(img_file)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

		self.gray = np.float32(self.gray)
		self.dst = cv2.cornerHarris(self.gray, 2, 3, 0.04)

		# result is dilated for marking the corners, not important
		self.dst = cv2.dilate(self.dst, None)

		# Threshold for an optimal value
		self.img[self.dst>0.01*self.dst.max()]=[0,0,255]
	
	def show_result(self):
		cv2.imshow('dst',self.img)
		if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()

# FAST returns the corners as a list of KeyPoints
class FAST(object):
	def __init__(self, img_file):
		self.img = cv2.imread(img_file)
		self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		fast = cv2.FastFeatureDetector_create()
		self.kp = fast.detect(self.gray,None)
		
	def show_result(self):
		img2 = cv2.drawKeypoints(self.img, self.kp, None, color=(255,0,0))
		cv2.imshow('kp',img2)
		if cv2.waitKey(0) & 0xff == 27:
			cv2.destroyAllWindows()
		
if __name__ == '__main__':
	img_file = 'imgs/training/4/6.png'
	#ch = CornerHarris(img_file)
	#ch.show_result()
	fast = FAST(img_file)
	fast.show_result()

