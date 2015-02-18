import os
from os import listdir
import pylab
import skimage
import skimage.io
import skimage.transform
import scipy.ndimage as ndimage
import os
import scipy
import matplotlib.pyplot as plt
from skimage.filter import roberts, sobel
from math import exp
import numpy as np
import math
import random


########################################################
############### CREATE DATABASE OF IMAGES ##$###########
########################################################

folder = os.getcwd()
images = {}
with open(os.path.join(folder, 'GroundTruth.txt')) as f:
	for line in f:
		l = line.split()
		if l[0] in images:
			images[l[0]].append(l[1:])
		else:
			images[l[0]] = []
			images[l[0]].append(l[1:])


faceCount = 0
nonFaceCount = 0
imageDB = []
allImages = os.path.join(os.getcwd(), 'allImages/')
for f in listdir(allImages):
	fullImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'allImages/')+f))
	if f in images:
		# for i in range(0,len(images[f])):
		if (faceCount < 100):
			lEx,lEy,rEx,rEy,nx,ny,lMx,lMy,cMx,cMy,rMx,rMy = images[f][0] 
			X = [float(lEx),float(rEx),float(nx),float(lMx),float(cMx),float(rMx)]
			Y = [float(lEy),float(rEy),float(ny),float(lMy),float(cMy),float(rMy)]

			boxRange = 30
			minY = min(Y) - boxRange
			maxY = max(Y) + boxRange
			minX = min(X) - boxRange
			maxX = max(X) + boxRange

			face = fullImg[minY:maxY,minX:maxX]
			try:
				face_Resized = scipy.misc.imresize(face,(12,12))
				imageDB.append(face_Resized)
				faceCount += 1
			except:
				pass

			

for f in listdir(allImages):
	fullImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'allImages/')+f))
	for i in range(0,10):
		if (nonFaceCount < 100):
			random.seed(i)
			x1 = random.randint(0,fullImg.shape[1])
			x2 = random.randint(0,fullImg.shape[1])

			y1 = random.randint(0,fullImg.shape[0])
			y2 = random.randint(0,fullImg.shape[0])

			minX = min(x1,x2)
			maxX = max(x1,x2)

			minY = min(y1,y2)
			maxY = max(y1,y2)

			notFace = fullImg[minY:maxY,minX:maxX]
			notFace_Resized = scipy.misc.imresize(notFace,(12,12))

			imageDB.append(notFace_Resized)
			nonFaceCount += 1

collage = np.concatenate(imageDB)
collage = np.reshape(collage, (120, 240))

skimage.io.imsave(folder + '/collage.png', collage)

imageDB = np.array(imageDB)


########################################################
########## GAUSSIAN DISTRIBUTION CLASSIFIER ############
########################################################


fMean = np.mean(imageDB[0:100])
nfMean = np.mean(imageDB[100:200])

newImageDB = np.ndarray(shape=(200,144), dtype=float, order='F')
for i in range(0,len(imageDB)):
	print i
	newImageDB[i] = imageDB[i].ravel()


A = newImageDB[0:100]

E = np.dot(np.transpose(A),A)


plt.plot(sorted(s, reverse=True)); #plt.show()

U, s, V = np.linalg.svd(E, full_matrices=True)



#plt.imshow(fullImg, cmap = plt.get_cmap('gray')); plt.show()
#plt.imshow(face, cmap = plt.get_cmap('gray')); plt.show()