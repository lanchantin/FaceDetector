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
################### GLOBAL FUNCTIONS ###################
########################################################
## Create Gaussian  Pyramids ##
def createGaussian(sigma):
    w = 1 + (int(6*sigma))
    G = np.zeros((w,w))
    k = 0
    for x in range(w):
        for y in range(w):
            G[x,y] = math.exp(-0.5 * ( math.pow((x-w/2)/sigma, 2.0) + math.pow((y-w/2)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
            k += G[x,y]
    for x in range(w):
        for y in range(w):
            G[x,y] /= k;
    return G

def createGaussianPyramid(inputImage):
	octaves = 5
	initialSigma = .3

	image = inputImage

	##### Gaussian Pyramid #####
	print 'Creating Gaussian Pyramid...\n'
	GaussPyramid = {}
	for octave in range(octaves):
	    sigma = initialSigma#*(2.0**(octave/float(3)))
	    gaussian = createGaussian(sigma)
	    GaussPyramid[octave] = scipy.signal.convolve2d(image,gaussian,boundary='symm',mode='same')
	    image = scipy.ndimage.interpolation.zoom(image,.75)

	return GaussPyramid

def checkBounds(img, x, y):
	if (x >= 0 and x < img.shape[0]) and ((y >= 0 and y < img.shape[1])):
		return True
	else:
		return False

def drawBox(img,i,j):
	boxWidth = 6
	for x in range(-boxWidth, boxWidth+1):
			for y in range(-boxWidth, boxWidth+1):
				if (x == boxWidth) or (y == boxWidth) or (x == -boxWidth) or (y == -boxWidth):
					try:
						img[x+i][y+j] = 1
					except:
						pass
	return img


########################################################
############### CREATE DATABASE OF IMAGES ##$###########
########################################################

def createDB(DIR):
	folder = DIR
	### Create Dictionary of all images with faces ###
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
	imageDB = []
	trainImages = os.path.join(os.getcwd(), 'trainImages/')
	for f in listdir(trainImages):
		if "png" in f:
			fullImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'trainImages/')+f))
			if f in images:
				# for i in range(0,len(images[f])):
				if (faceCount < 100):
					lEx,lEy,rEx,rEy,nx,ny,lMx,lMy,cMx,cMy,rMx,rMy = images[f][0]
					X = [float(lEx),float(rEx),float(nx),float(lMx),float(cMx),float(rMx)]
					Y = [float(lEy),float(rEy),float(ny),float(lMy),float(cMy),float(rMy)]

					boxRange = 9
					minY = min(Y) - boxRange
					maxY = max(Y) + boxRange
					minX = min(X) - boxRange
					maxX = max(X) + boxRange

					face = fullImg[minY:maxY,minX:maxX]

					yRatio = float(12)/float(face.shape[0])
					xRatio = float(12)/float(face.shape[1])
					face_Resized = scipy.ndimage.interpolation.zoom(face,(yRatio,xRatio))

					imageDB.append(face_Resized)
						#skimage.io.imsave(folder + '/faces/image' + str(faceCount) + '.png', face_Resized)
					plt.imsave(folder + '/faces/image' + str(faceCount) + '.png', face_Resized, cmap = plt.get_cmap('gray'))
					faceCount += 1

	#plt.imshow(imageDB[0], cmap = plt.get_cmap('gray')); plt.show()
	nonFaceCount = 0

	for f in listdir(trainImages):
		random.seed(1)
		if "gif" in f:
			fullImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'trainImages/')+f, as_grey=True))
			if f.replace("gif","png") in images:
				pass
			else:
				#print f
				for i in range(0,8):
					if (nonFaceCount < 100):
						x1 = random.randint(0,fullImg.shape[1]-12)
						x2 = x1 + 12

						y1 = random.randint(0,fullImg.shape[0]-12)
						y2 = y1 + 12

						minX = min(x1,x2)
						maxX = max(x1,x2)

						minY = min(y1,y2)
						maxY = max(y1,y2)

						notFace = fullImg[minY:maxY,minX:maxX]

						yRatio = float(12)/float(notFace.shape[0])
						xRatio = float(12)/float(notFace.shape[1])
						notFace_Resized = scipy.ndimage.interpolation.zoom(notFace,(yRatio,xRatio))

						imageDB.append(notFace_Resized)

						plt.imsave(folder + '/nonFaces/image' + str(nonFaceCount) + '.png', notFace_Resized, cmap = plt.get_cmap('gray'))
						#skimage.io.imsave(folder + '/nonFaces/image' + str(nonFaceCount) + '.png', notFace_Resized)

						nonFaceCount += 1

	imageDB = np.array(imageDB)

	return imageDB



def createCollage(iDB, DIR):
	folder = DIR

	resizeImgDB = scipy.ndimage.interpolation.zoom(iDB,(1,2,2))


	collage = np.ndarray(shape=(240,480), dtype=np.dtype(np.float64))
	k = 0
	for i in range(0,240,24):
		for j in range(0,240,24):
			collage[i:i+24,j:j+24] = resizeImgDB[k]
			k+=1

	k = 100
	for i in range(0,240,24):
		for j in range(240,480,24):
			collage[i:i+24,j:j+24] = resizeImgDB[k]
			k+=1


	plt.imshow(collage, cmap = plt.get_cmap('gray')); #plt.show()
	plt.savefig(folder + '/collage.png')

	return collage


def findMeans(iDB):
	imageDB = iDB
	posMean = np.ndarray(shape=(12,12))
	negMean = np.ndarray(shape=(12,12))
	for i in range(0,12):
		for j in range(0,12):
			posMean[i,j] = np.mean(imageDB[0:100,i,j])
			negMean[i,j] = np.mean(imageDB[100:200,i,j])

	#skimage.io.imsave(folder + '/meanFace.png', posMean)
	plt.imshow(posMean, cmap = plt.get_cmap('gray')); #plt.show()
	plt.savefig(folder + '/meanPosFace.png')

	plt.imshow(negMean, cmap = plt.get_cmap('gray')); #plt.show()
	plt.savefig(folder + '/meanNegFace.png')

	return (posMean, negMean)

#############################
folder = folder = os.getcwd()
imageDB = createDB(folder)
collage = createCollage(imageDB, folder)
posMean, negMean = findMeans(imageDB)

##############################


##############################################################################################
##############################################################################################
############################## GAUSSIAN DISTRIBUTION CLASSIFIER ##############################
##############################################################################################
##############################################################################################

##### Create means ######


def ComputeSVD(A, threshold):
	E = np.dot(np.transpose(A),A)

	U, s, V = np.linalg.svd(E, full_matrices=True)

	#print s
	#plt.plot(s); plt.show()

	listElements = s[0: int(len(s) * .1)]

	Uk = []
	sk = []
	k = 0
	for i in range(0,len(listElements)):
		k+=1
		Uk.append(U[i])
		sk.append(s[i])

	# Uk = []
	# sk = []
	# k = 0
	# for i in range(0,len(s)):
	# 	if s[i] > threshold:
	# 		k+=1
	# 		Uk.append(U[i])
	# 		sk.append(s[i])

	Uk = np.array(Uk)
	Uk = np.transpose(Uk)

	sk = np.array(sk)
	Sk = np.diag(sk)

	Ek = np.dot(np.dot(Uk,Sk),np.transpose(Uk))

	return (Ek, Uk, Sk, k)


def ComputeG(x, Uk, Sk, mean,k):
	denom = ( (math.pow(2*math.pi, (k/2)))*(math.sqrt(np.linalg.det(Sk))) )
	C = 1 / denom

	TEMP1 = np.dot(np.transpose(x-mean),Uk)
	TEMP2 = np.dot(TEMP1,np.linalg.inv(Sk))
	TEMP3 = np.dot(np.transpose(Uk),(x-mean))
	TEMP4 = np.dot(TEMP3, TEMP2)

	EXP = math.exp(float((-0.5)*TEMP4))

	G = C*EXP

	return (G, C)


imageDB144 = np.ndarray(shape=(200,144), dtype=float, order='F')
for i in range(0,len(imageDB)):
	imageDB144[i] = imageDB[i].ravel()


A_Pos = []
for i in range(0,100):
	A_Pos.append(imageDB144[i] - posMean.ravel())
A_Pos = np.array(A_Pos)

A_Neg = []
for i in range(100,200):
	A_Neg.append(imageDB144[i] - negMean.ravel())
A_Neg = np.array(A_Neg)


thres = 1.24007132e+00

############## POSITIVE GAUSSIAN ################
Ek_Pos, Uk_Pos, Sk_Pos, k_Pos = ComputeSVD(A_Pos, thres)


############## NEGATIVE GAUSSIAN ################
Ek_Neg, Uk_Neg, Sk_Neg, k_Neg = ComputeSVD(A_Neg, thres)


totalFaces = 0
pi = .99
for i in range(0,100):
	x = imageDB144[i]



	print 'G_POS:',(G_POS*pi)

	print 'G_NEG:', (G_NEG*(1-pi))

	CLASSIFIER = (G_POS*pi)/(G_NEG*(1-pi))

	if CLASSIFIER > 1:
		totalFaces += 1

	print 'CLASSIFIER:',CLASSIFIER

	print ""




def findFacesGaussian(inputImage):
	image = inputImage

	mask = np.ndarray(shape=(image.shape[0],image.shape[1]), dtype=np.dtype(np.float64))
	for i in range(0,mask.shape[0]):
		for j in range(0,mask.shape[1]):
			mask[i,j] = 0

	for i in range(0,(image.shape[0]-12)):
		for j in range(0,(image.shape[1]-12)):
			currFrame = image[i:i+12,j:j+12]
			x = [0]*145
			x[0:144] = currFrame.ravel()
			x[144] = 1
			
			G_POS, C_POS = ComputeG(x,Uk_Pos, Sk_Pos, posMean.ravel(), k_Pos)
			G_NEG, C_NEG = ComputeG(x,Uk_Neg, Sk_Neg, negMean.ravel(), k_Neg)

			pi = 0.5
			CLASSIFIER = (G_POS*pi)/(G_NEG*(1-pi))
			if CLASSIFIER > 1:
				mask[i,j] = G_POS
	# Nonmaximum Suppresion
	M = 10
	for i in range(0,(image.shape[0]-12)):
		for j in range(0,(image.shape[1]-12)):
			for k in range(-M,M,1):
				for r in range(-M,M,1):
					if checkBounds(image, i+k,j+r):
						if mask[i,j] >  mask[i+k,j+r]:
							mask[i+k,j+r] = 0;


	for i in range(0,mask.shape[0]):
		for j in range(0,mask.shape[1]):
			if mask[i,j] > 0:
				image = drawBox(image,i+6,j+6)

	return image


testImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'testImages/married.png')))

GaussianPyramid = None
GaussianPyramid = createGaussianPyramid(testImg)

extractedFaces = {}
for i in GaussianPyramid:
	#plt.imshow(GaussPyramid[i], cmap = plt.get_cmap('gray')); plt.show()
	extractedFaces[i] = findFacesGaussian(GaussianPyramid[i])









######################################################################################################
######################################################################################################
######################################################################################################
####################################### LOGISTIC REGRESSION ##########################################
######################################################################################################
######################################################################################################
######################################################################################################


X = np.ndarray(shape=(200,145), dtype=np.dtype(np.float64))

Y = np.zeros(200)
Y[0:100] = 1

for i in range(0,len(imageDB)):
	X[i,0:144] = imageDB[i].ravel()
	X[i,144] = 1

global w
w = np.zeros(145,dtype=np.dtype(np.float64))

for j in range(0,10000):
	sum1 = 0
	for i in range(0,200):
		g = 1/float(1+np.exp(-(np.dot(np.transpose(w),X[i]))))
		sum1 += np.dot((Y[i] - g),X[i])

	n = 0.5 #learning rate
	w = w + n*sum1


# Test train data (sanity check)
# for i in range(0,200):
# 	g = 1/float(1+np.exp(-(np.dot(np.transpose(w),X[i]))))
# 	print Y[i]
# 	print g
# 	print ""

#############################################################
####################### TEST FUNCTIONS ######################
#############################################################


def findFacesLogistic(inputImage):
	image = inputImage

	mask = np.ndarray(shape=(image.shape[0],image.shape[1]), dtype=np.dtype(np.float64))
	for i in range(0,mask.shape[0]):
		for j in range(0,mask.shape[1]):
			mask[i,j] = 0

	for i in range(0,(image.shape[0]-12)):
		for j in range(0,(image.shape[1]-12)):
			currFrame = image[i:i+12,j:j+12]
			x = [0]*145
			x[0:144] = currFrame.ravel()
			x[144] = 1
			g = 1/float(1+np.exp(-(np.dot(w,x))))
			#print np.dot(w,x)
			if (g > 0.5):
				mask[i,j] = np.dot(w,x)

	# Nonmaximum Suppresion
	M = 10
	for i in range(0,(image.shape[0]-12)):
		for j in range(0,(image.shape[1]-12)):
			for k in range(-M,M,1):
				for r in range(-M,M,1):
					if checkBounds(image, i+k,j+r):
						if mask[i,j] >  mask[i+k,j+r]:
							mask[i+k,j+r] = 0;


	for i in range(0,mask.shape[0]):
		for j in range(0,mask.shape[1]):
			if mask[i,j] > 0:
				image = drawBox(image,i+6,j+6)

	return image

#############################################################
###################### TEST ON IMAGES #######################
#############################################################

#plt.imshow(testImg, cmap = plt.get_cmap('gray')); plt.show()
testImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'testImages/married.png')))

GaussianPyramid = None
GaussianPyramid = createGaussianPyramid(testImg)

extractedFaces = {}
for i in GaussianPyramid:
	#plt.imshow(GaussPyramid[i], cmap = plt.get_cmap('gray')); plt.show()
	extractedFaces[i] = findFaces(GaussianPyramid[i])

for i in extractedFaces:
	plt.imshow(extractedFaces[i], cmap = plt.get_cmap('gray')); plt.show()

collage = scipy.ndimage.interpolation.zoom(collage,.5)
ti2 = collage
f = findFaces(ti2)
plt.imshow(f, cmap = plt.get_cmap('gray')); plt.show()

findFacesLogistic(GaussianPyramid[2])
plt.imshow(GaussianPyramid[1], cmap = plt.get_cmap('gray')); plt.show()
