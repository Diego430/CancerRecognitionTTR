import numpy as np
import random
from pca import pcafn

import matplotlib.pyplot as plt



def loadCancerDataset(samplepath, lablepath):
	"""
	Importo il dataset contenuto nel file npy in samplepath con le relative lable nel file npy lablepath
	Args:
		samplepath: path del file npy contenente sample e features
		lablepath: path del file npy contenente lable relative al file dei sample
		traintest: percentuale di suddivisione train / test (0.8 = 80% train, 20% test)

	Returns:
		xtrain, ytrain, xtest, ytest
	"""
	# carico il dataset da file npy
	samples = np.load(samplepath)
	lables = np.load(lablepath)

	return samples, lables

def loadCancerDatasetPCA(samplepath, lablepath):
	
	"""
	Importo il dataset contenuto nel file npy in samplepath con le relative lable nel file npy lablepath
	e applico la pca prima di ritornarlo
	Args:
		samplepath: path del file npy contenente sample e features
		lablepath: path del file npy contenente lable relative al file dei sample
		traintest: percentuale di suddivisione train / test (0.8 = 80% train, 20% test)

	Returns:
		samples, lables, nfeatures
	"""
	# carico il database non appiattito
	samples, lables = loadCancerDataset(samplepath, lablepath)
	
	# massimizzo il numero di features estraibili concesse dal PCA: min(nsamples, nfeatures)
	nfeatures = min(samples.shape[0], samples.shape[1]) 

	# appiattisco con pca i samples di train e test
	samples = pcafn(samples, nfeatures)

	return samples, lables

def loadCancerDatasetShuffle(samplepath, lablepath, traintest=0.8):
	"""
	Importo il dataset contenuto nel file npy in samplepath con le relative lable nel file npy lablepath
	ordino samples per classe in modo random prima di ritornarlo

	Args:
		samplepath: path del file npy contenente sample e features
		lablepath: path del file npy contenente lable relative al file dei sample
		traintest: percentuale di suddivisione train / test (0.8 = 80% train, 20% test)

	Returns:
		xtrain, ytrain, xtest, ytest
	"""
	# carico il dataset da file npy
	samples = np.load(samplepath)
	lables = np.load(lablepath)

	# estraggo le classi
	classes = np.unique(lables)
	nclasses = classes.shape[0]
	
	# preparo array di train e test
	xtrain = np.empty((0, samples.shape[1]))
	ytrain = np.empty(0)
	xtest = np.empty((0, samples.shape[1]))
	ytest = np.empty(0)

	for classname in classes:
		# estraggo i campioni per classe
		classindex = np.where(lables == classname)
		classsamples = samples[classindex]
		classlables = lables[classindex]
		
		# creo punto di divisione
		divisione = int(classsamples.shape[0] * traintest)

		# scombino in modo random i campioni per classe
		random.shuffle(classsamples)
		
		# aggiungo i campioni randomizzati agli array di train e test 
		xtrain = np.concatenate((xtrain,classsamples[:divisione]), 0)
		xtest = np.concatenate((xtest, classsamples[divisione:]), 0)
		ytrain = np.concatenate((ytrain, classlables[:divisione]), 0)
		ytest = np.concatenate((ytest, classlables[divisione:]), 0)

	return xtrain, ytrain, xtest, ytest

def loadCancerDatasetPCAShuffle(samplepath, lablepath, traintest=0.8):
	
	"""
	Importo il dataset contenuto nel file npy in samplepath con le relative lable nel file npy lablepath
	ordino samples per classe in modo random e applico la pca prima di ritornarlo
	Args:
		samplepath: path del file npy contenente sample e features
		lablepath: path del file npy contenente lable relative al file dei sample
		traintest: percentuale di suddivisione train / test (0.8 = 80% train, 20% test)

	Returns:
		xtrain, ytrain, xtest, ytest
	"""
	# carico il database non appiattito
	xtrain, ytrain, xtest, ytest = loadCancerDatasetShuffle(samplepath, lablepath, traintest=0.8)
	
	# massimizzo il numero di features estraibili concesse dal PCA: min(nsamples, nfeatures)
	nfeatures = min(xtrain.shape[0], xtrain.shape[1], xtest.shape[0], xtest.shape[1]) 

	# appiattisco con pca i samples di train e test
	xtrain = pcafn(xtrain, nfeatures)
	xtest = pcafn(xtest, nfeatures)

	return xtrain, ytrain, xtest, ytest

def pcaplot():
	"""
	Mostro lo scatter plot del pca di ogni dataset
	"""
	sampledata = ["cancer/brainSamples.npy", "cancer/leukemiaSamples.npy", "cancer/liverSamples.npy"] 
	labledata = ["cancer/brainLable.npy", "cancer/leukemiaLable.npy", "cancer/liverLable.npy"]
	for dataset in range(len(sampledata)):
		samples, lables = loadCancerDataset(sampledata[dataset], labledata[dataset])
		nfeatures = 2 
		samples = pcafn(samples, nfeatures)
		classes = np.unique(lables)
		fig, ax = plt.subplots()
		for color in range(classes.shape[0]):
			indexes = np.where(lables == classes[color])[0]
			head = indexes[0]
			tail = indexes[-1]
			x = samples[head:tail,0]
			y = samples[head:tail,1]
			ax.scatter(x, y, label=classes[color])

		ax.legend()
		ax.grid(True)
		plt.title(sampledata[dataset].split('/')[1].split('Samples')[0])
		plt.show()
