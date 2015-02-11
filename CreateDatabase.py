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

images = {}
with open(os.path.join(os.getcwd(), 'GroundTruth.txt')) as f:
	for line in f:
		l = line.split()
		if l[0] in images:
			images[l[0]].append(l[1:])
		else:
			images[l[0]] = []
			images[l[0]].append(l[1:])


imageDB = []
allImages = os.path.join(os.getcwd(), 'allImages/')
for f in listdir(allImages):
	fullImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'allImages/')+f))
	if f in images:
		lEx,lEy,rEx,rEy,nx,ny,lMx,lMy,cMx,cMy,rMx,rMy = images[f][0] 
		X = [float(lEx),float(rEx),float(nx),float(lMx),float(cMx),float(rMx)]
		Y = [float(lEy),float(rEy),float(ny),float(lMy),float(cMy),float(rMy)]

		minY = min(Y) - 10
		maxY = max(Y) + 10
		minX = min(X) - 10
		maxX = max(X) + 10

		face = fullImg[minY:maxY,minX:maxX]
		face_Resized = scipy.misc.imresize(face,(12,12))
		imageDB.append(face_Resized)
	else:
		random.seed(64)
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


#plt.imshow(fullImg, cmap = plt.get_cmap('gray')); plt.show()
#plt.imshow(face, cmap = plt.get_cmap('gray')); plt.show()