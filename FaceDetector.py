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

					yRatio = float(12)/float(notFace.shape[0])
					xRatio = float(12)/float(notFace.shape[1])
					notFace_Resized = scipy.ndimage.interpolation.zoom(notFace,(yRatio,xRatio))

					imageDB.append(notFace_Resized)

					plt.imsave(folder + '/nonFaces/image' + str(nonFaceCount) + '.png', notFace_Resized, cmap = plt.get_cmap('gray'))
					#skimage.io.imsave(folder + '/nonFaces/image' + str(nonFaceCount) + '.png', notFace_Resized)
					
					nonFaceCount += 1

imageDB = np.array(imageDB)


# collage = np.concatenate(imageDB)
# collage = np.reshape(collage, (120, 240))

# plt.imshow(collage, cmap = plt.get_cmap('gray')); #plt.show()
# plt.savefig(folder + '/collage.png')



########################################################
########## GAUSSIAN DISTRIBUTION CLASSIFIER ############
########################################################

### Create means ###
fMean = np.ndarray(shape=(12,12))
nfMean = np.ndarray(shape=(12,12))
for i in range(0,12):
	for j in range(0,12):
		fMean[i,j] = np.mean(imageDB[0:100,i,j])
		nfMean[i,j] = np.mean(imageDB[100:200,i,j])

#skimage.io.imsave(folder + '/meanFace.png', fMean)
plt.imshow(fMean, cmap = plt.get_cmap('gray')); #plt.show()
plt.savefig(folder + '/meanFace.png')

plt.imshow(nfMean, cmap = plt.get_cmap('gray')); #plt.show()
plt.savefig(folder + '/meanNonFace.png')


newImageDB = np.ndarray(shape=(200,144), dtype=float, order='F')
for i in range(0,len(imageDB)):
	newImageDB[i] = imageDB[i].ravel()

Apos = []
for i in range(0,100):
	Apos.append(newImageDB[i] - fMean.ravel())
Apos = np.array(Apos)



E = np.dot(np.transpose(Apos),Apos)

U, s, V = np.linalg.svd(E, full_matrices=True)

thresh = 1.52920728e+00

Uk = []
sk = []
k = 0
for i in range(0,10):
		k+=1
		Uk.append(U[i])
		sk.append(s[i])  
# for i in range(0,len(U)):
# 	# print s[i]
# 	if s[i] > thresh:
# 		k+=1
# 		Uk.append(U[i])
# 		sk.append(s[i])

Uk = np.array(Uk)
Uk = np.transpose(Uk)


sk = np.array(sk)
Sk = np.diag(sk)

Ek = np.dot(np.dot(Uk,Sk),np.transpose(Uk))




C = 1/ ( (math.pow(2*math.pi, (k/2))) * (math.sqrt(np.linalg.det(Sk))))

G_POS_ARR = []
for i in range(0,100):
	x = newImageDB[i].ravel()
	mean = fMean.ravel()
	TEMP1 = np.dot(np.transpose(x-mean),Uk)
	TEMP2 = np.dot(TEMP1,np.linalg.inv(Sk))
	TEMP3 = np.dot(np.transpose(Uk),(x-mean))
	TEMP4 = np.dot(TEMP3, TEMP2)

	EXP = math.exp(float((-0.5)*TEMP4))

	G_POS = C*EXP
	G_POS_ARR.append(G_POS)
	print G_POS


print ""
print ""

A_NEG = []
for i in range(0,100):
	A_NEG.append(newImageDB[i] - nfMean.ravel())
A_NEG = np.array(A_NEG)


E = np.dot(np.transpose(A_NEG),A_NEG)

U, s, V = np.linalg.svd(E, full_matrices=True)

thresh = 0
Uk = []
sk = []
k = 0
for i in range(0,10):
		k+=1
		Uk.append(U[i])
		sk.append(s[i])  

Uk = np.array(Uk)
Uk = np.transpose(Uk)


sk = np.array(sk)
Sk = np.diag(sk)

Ek = np.dot(np.dot(Uk,Sk),np.transpose(Uk))



C = 1/ ( (math.pow(2*math.pi, (k/2))) * (math.sqrt(np.linalg.det(Sk))))

G_NEG_ARR = []
for i in range(0,100):
	x = newImageDB[i].ravel()
	mean = nfMean.ravel()
	TEMP1 = np.dot(np.transpose(x-mean),Uk)
	TEMP2 = np.dot(TEMP1,np.linalg.inv(Sk))
	TEMP3 = np.dot(np.transpose(Uk),(x-mean))
	TEMP4 = np.dot(TEMP3, TEMP2)

	EXP = math.exp(float((-0.5)*TEMP4))

	G_NEG = C*EXP
	G_NEG_ARR.append(G_NEG)
	print G_NEG

k = 0
for i in range(0,100):
	if G_POS_ARR[i] > G_NEG_ARR[i]:
		k+=1


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
		g = 1/float(1+np.exp(-(np.dot(w,X[i]))))
		sum1 += np.dot((Y[i] - g),X[i])


	n = 0.5 #learning rate
	w = w + n*sum1



testImg = skimage.img_as_float(skimage.io.imread(os.path.join(os.getcwd(), 'testImages/married.gif')))




for i in range(0,(testImg.shape[0]-12),12):
	for j in range(0,(testImg.shape[1]-12),12):
		currFrame = testImg[i:i+12,j:j+12]
		x[0:144] = currFrame.ravel()
		x[144] = 1
		g = 1/float(1+np.exp(-(np.dot(w,x))))
		print g
		if (g > 0.5):
			for k in range(i,i+12):
				for l in range(j,j+12):
					if ((k == i) or (k == i-12) or (l == j) or (l == j-12)):
						testImg[k,l] = 1


plt.imshow(testImg, cmap = plt.get_cmap('gray')); plt.show()	

skimage.io.imshow(newTestImg);skimage.io.show()





#plt.imshow(fullImg, cmap = plt.get_cmap('gray')); plt.show()
