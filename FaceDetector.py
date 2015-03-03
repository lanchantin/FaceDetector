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

				boxRange = 10
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
	if "gif" in f:
		fullImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'trainImages/')+f, as_grey=True))
		if f.replace("gif","png") in images:
			pass
		else:
			print f
			for i in range(0,8):
				if (nonFaceCount < 100):
					random.seed(i)
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

resizeImgDB = scipy.ndimage.interpolation.zoom(imageDB,(1,2,2))


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


########################################################
########## GAUSSIAN DISTRIBUTION CLASSIFIER ############
########################################################
##### Create means ######
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


def ComputeSVD(A):
	E = np.dot(np.transpose(A),A)

	U, s, V = np.linalg.svd(E, full_matrices=True)

	thres = 1.52920728e+00

	Uk = []
	sk = []
	k = 0
	for i in range(0,threshold):
		k+=1
		Uk.append(U[i])
		sk.append(s[i]) 

	Uk = np.array(Uk)
	Uk = np.transpose(Uk)

	sk = np.array(sk)
	Sk = np.diag(sk)

	Ek = np.dot(np.dot(Uk,Sk),np.transpose(Uk))

	return (Ek, Uk, Sk)


def ComputeG(x, Uk, Sk, mean):
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


threshold = 10
k = threshold





############## POSITIVE GAUSSIAN ################
Ek_Pos, Uk_Pos, Sk_Pos = ComputeSVD(A_Pos)

G_POS_ARR = []
for i in range(0,100):
	x = imageDB144[i]

	G_POS, C_POS = ComputeG(x,Uk_Pos, Sk_Pos, posMean.ravel())

	G_POS_ARR.append(G_POS)

	print G_POS


print "" ; print ""


############## NEGATIVE GAUSSIAN ################
Ek_Neg, Uk_Neg, Sk_Neg = ComputeSVD(A_Neg)

G_NEG_ARR = []
for i in range(0,100):
	x = imageDB144[i].ravel()

	G_NEG, C_NEG = ComputeG(x,Uk_Neg, Sk_Neg, negMean.ravel())

	G_NEG_ARR.append(G_NEG)

	print G_NEG



kF = 0
for i in range(0,100):
	if G_POS_ARR[i] > G_NEG_ARR[i]:
		kF+=1

kNF = 0
for i in range(0,100):
	if G_POS_ARR[i] < G_NEG_ARR[i]:
		kNF+=1

# plt.plot(sorted(s, reverse=True)); plt.show()

########################################################
################### LOGISTIC REGRESSION ################
########################################################

X = np.ndarray(shape=(200,145), dtype=np.dtype(np.float64))

Y = np.zeros(200)
Y[0:100] = 1

for i in range(0,len(imageDB)):
	X[i,0:144] = imageDB[i].ravel()
	X[i,144] = 1

w = np.zeros(145,dtype=np.dtype(np.float64))

for j in range(0,10000):
	sum1 = 0
	for i in range(0,200): 
		g = 1/float(1+np.exp(-(np.dot(np.transpose(w),X[i]))))
		sum1 += np.dot((Y[i] - g),X[i])

	n = 0.5 #learning rate
	w = w + n*sum1


testImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'testImages/married.png')))



	for i in range(0,200): 
		g = 1/float(1+np.exp(-(np.dot(np.transpose(w),X[i]))))
		print Y[i]
		print g


for i in range(0,(testImg.shape[0]-12)):
	for j in range(0,(testImg.shape[1]-12)):
		currFrame = testImg[i:i+12,j:j+12]
		x = [0]*145
		x[0:144] = currFrame.ravel()
		x[144] = 1
		g = 1/float(1+np.exp(-(np.dot(w,x))))
		#print g
		if (g > 0.5):
			for k in range(i,i+12):
				for l in range(j,j+12):
					if ((k == i) or (k == i-12) or (l == j) or (l == j-12)):
						testImg[k,l] = 1


plt.imshow(testImg, cmap = plt.get_cmap('gray')); plt.show()	


#plt.imshow(fullImg, cmap = plt.get_cmap('gray')); plt.show()




###########################################################
################ Create Gaussian  Pyramids ################
###########################################################
octaves = 4
scales = 5
initialSigma = 1.6


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


##### Gaussian Pyramid #####
print 'Creating Gaussian Pyramid...\n'
testImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'testImages/married.png')))
GaussPyramid = {}
for octave in range(octaves):
    sigma = initialSigma*(2.0**(octave/float(3)))
    gaussian = createGaussian(sigma)
    GaussPyramid[octave] = scipy.signal.convolve2d(testImg,gaussian,boundary='symm',mode='same')
    testImg = scipy.ndimage.interpolation.zoom(testImg,.5)
