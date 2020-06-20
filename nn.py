"""
cambiare output 
cambiare funzione di loss

"""

import torch
from torch import nn
from utility import loadCancerDatasetPCAShuffle, loadCancerDatasetShuffle
import numpy as np
import random


verbose=False
debug=False

def rawnnfn(dataset=0, usepca=True, traintest = 0.8, maxepoch=25000, learningrate=1e-8, devicename='cpu'):
	"""
	Implemento il riconoscimento usando NN in modo raw

	Args:
		dataset: indica il dataset da caricare (0 : brain, 1 : leukemia, 2 : liver)
		usepca: True per usare database con dimensionalità ridotta dall PCA, False senza PCA
		traintest: percentuale di suddivisione train / test (0.8 = 80% train, 20% test)
		maxepoch: numero massimo di epoch 
		learningrate: il learning rate che deve avere la nn (se >= 1e-5 losstrain diventa nan)
		devicename: il device da far usare a pytorch per istanziare i tensor 
	Returns:
		cmc: confusion matrix
		accuracy: (veri positivi + veri negativi) / totale
		precision: veri positivi / (veri positivi + falsi positivi)
		recall: veri positivi / (veri positivi + falsi negativi) 
		nfeatures: numero delle features usate per il riconoscimento 
	"""

	sampledata = ["cancer/brainSamples.npy", "cancer/leukemiaSamples.npy", "cancer/liverSamples.npy"] 
	labledata = ["cancer/brainLable.npy", "cancer/leukemiaLable.npy", "cancer/liverLable.npy"]
	dtype = torch.float
	device = torch.device(devicename)

	if verbose:
		print("Loading dataset...")
	
	# Carico dataset
	if usepca:
		xtrain, ytrain, xtest, ytest = loadCancerDatasetPCAShuffle(sampledata[dataset], labledata[dataset], traintest)	
	else:
		xtrain, ytrain, xtest, ytest = loadCancerDatasetShuffle(sampledata[dataset], labledata[dataset], traintest)	
	
	# classi presenti nel database, sono ordinate in ordine alfabetico 
	classes = np.unique(ytrain)
	nclasses = classes.shape[0]
	ntrainsamples = xtrain.shape[0]
	ntestsamples = xtest.shape[0]
	nfeatures = xtrain.shape[1]
	nhidden = nclasses

	# converto in interi le lable 
	newytrain = torch.tensor((), device=device, dtype=dtype)
	newytrain = newytrain.new_zeros((ntrainsamples,nclasses))
	for l in range(ytrain.shape[0]):
		lableindex = int(np.where(classes == ytrain[l])[0][0])
		newytrain[l][lableindex] = 1

	newytest = torch.tensor((), device=device, dtype=dtype)
	newytest = newytest.new_zeros((ntestsamples, nclasses))
	for l in range(ytest.shape[0]):
		lableindex = np.where(classes == ytest[l])[0][0]
		newytest[l][lableindex] = 1
		
	xtrain = torch.from_numpy(xtrain).type(dtype)
	ytrain = newytrain
	xtest = torch.from_numpy(xtest).type(dtype)
	ytest = newytest

	# Inizializzo i pesi randomicamente
	w1 = torch.randn(nfeatures, nhidden, device=device, dtype=dtype, requires_grad=True)
	w2 = torch.randn(nhidden, nclasses, device=device, dtype=dtype, requires_grad=True)

	b1 = torch.randn(nhidden, device=device, dtype=dtype, requires_grad=True)
	b2 = torch.randn(nclasses, device=device, dtype=dtype, requires_grad=True)


	if verbose:
		print("Training...")

	for epoch in range(maxepoch):
		
		# Passo di forward:
		hidden = xtrain.mm(w1) - b1 
		hrelu = hidden.clamp(min=0) 
		ypredict = hrelu.mm(w2) - b2
		

		# Calcolo la loss
		difference = (ypredict - ytrain)
		differencepow = difference.pow(2)*.5
		losstrain = differencepow.sum()

		# Ottengo le predizioni dei casi di test
		with torch.no_grad():
			predtest = (xtest.mm(w1) - b1).clamp(min=0).mm(w2) - b2

		# Faccio il plot delle predizioni rispetto ai valori di ground-truth
		if verbose and epoch%1000 == 0:
			print(f"epoch {epoch}: {losstrain.item()}")
			
		# Backprop to compute gradients of w1 and w2 with respect to loss
		losstrain.backward()

		with torch.no_grad():
			w1 -= learningrate * w1.grad
			w2 -= learningrate * w2.grad

			b1 -= learningrate * b1.grad
			b2 -= learningrate * b2.grad

			# Manually zero the gradients after updating weights
			w1.grad.zero_()
			w2.grad.zero_()

			b1.grad.zero_()
			b2.grad.zero_()
	
	# predtest contiene le predict della nn sui casi di test
	print(predtest)
	print(ytest)

def nnfn(dataset=0, usepca=True, traintest = 0.8, maxepoch=25000, learningrate=1e-8, devicename='cpu'):
	"""
	Implemento il riconoscimento usando NN di Pytorch

	Args:
		dataset: indica il dataset da caricare (0 : brain, 1 : leukemia, 2 : liver)
		usepca: True per usare database con dimensionalità ridotta dall PCA, False senza PCA
		traintest: percentuale di suddivisione train / test (0.8 = 80% train, 20% test)
		maxepoch: numero massimo di epoch 
		learningrate: il learning rate che deve avere la nn (se >= 1e-5 losstrain diventa nan)
		devicename: il device da far usare a pytorch per istanziare i tensor 
	Returns:
		cmc: confusion matrix
		accuracy: (veri positivi + veri negativi) / totale
		precision: veri positivi / (veri positivi + falsi positivi)
		recall: veri positivi / (veri positivi + falsi negativi) 
		nfeatures: numero delle features usate per il riconoscimento 
	"""

	sampledata = ["cancer/brainSamples.npy", "cancer/leukemiaSamples.npy", "cancer/liverSamples.npy"] 
	labledata = ["cancer/brainLable.npy", "cancer/leukemiaLable.npy", "cancer/liverLable.npy"]
	dtype = torch.float
	device = torch.device(devicename)

	if verbose:
		print("Loading dataset...")
	
	# Carico dataset
	if usepca:
		xtrain, ytrain, xtest, ytest = loadCancerDatasetPCAShuffle(sampledata[dataset], labledata[dataset], traintest)	
	else:
		xtrain, ytrain, xtest, ytest = loadCancerDatasetShuffle(sampledata[dataset], labledata[dataset], traintest)	
	
	# classi presenti nel database, sono ordinate in ordine alfabetico 
	classes = np.unique(ytrain)
	nclasses = classes.shape[0]
	ntrainsamples = xtrain.shape[0]
	ntestsamples = xtest.shape[0]
	nfeatures = xtrain.shape[1]
	nhidden = nclasses

	# converto in interi le lable 
	newytrain = torch.tensor((), device=device, dtype=dtype)
	newytrain = newytrain.new_zeros((ntrainsamples,nclasses))
	for l in range(ytrain.shape[0]):
		lableindex = int(np.where(classes == ytrain[l])[0][0])
		newytrain[l][lableindex] = 1

	newytest = torch.tensor((), device=device, dtype=dtype)
	newytest = newytest.new_zeros((ntestsamples,nclasses))
	for l in range(ytest.shape[0]):
		lableindex = np.where(classes == ytest[l])[0][0]
		newytest[l][lableindex] = 1
		
	xtrain = torch.from_numpy(xtrain).type(dtype)
	ytrain = newytrain
	xtest = torch.from_numpy(xtest).type(dtype)
	ytest = newytest

	# Creo il modello per sostiture la parte di forward
	model = nn.Sequential(
		nn.Linear(nfeatures,nhidden),
		nn.ReLU(),
		nn.Linear(nhidden, nclasses)
	)

	# definisco la loss con mean squared error (reduction = 'mean' or 'sum')
	lossfn = nn.MSELoss(reduction='sum')

	# definisco l'ottimizzatore
	optimizer = torch.optim.SGD(model.parameters(), lr=learningrate)

	if verbose:
		print("Training...")
	
	for epoch in range(maxepoch):
		# Passo di forward
		ypredict = model(xtrain)

		# Calcolo la loss
		losstrain = lossfn(ypredict, ytrain)

		# Ottengo le predizioni dei casi di test
		with torch.no_grad():
			predtest = model(xtest)

		# Faccio il plot delle predizioni rispetto ai valori di ground-truth
		if verbose and epoch%1000 == 0:
			print(epoch, losstrain.item())

		# azzero il gradiente delle variabili da aggiornare
		optimizer.zero_grad()

		# computo il gradiente della loss function (passo di backward)
		losstrain.backward()

		# aggiorno i parametri dell'ottimizzatore
		optimizer.step()

	# predtest contiene le predict della nn sui casi di test

	predlables = torch.tensor((), device=device, dtype=dtype)
	predlables = predlables.new_zeros((ntestsamples,nclasses))
	for l in range(predtest.shape[0]):
		lableindex = predtest[l].max(0)[1].item()
		predlables[l][lableindex] = 1
	

	# Creo la confusion matrix
	cmc = np.zeros((nclasses, nclasses))
	for predict, testlable in zip(predlables, ytest):
		# recupero indice della classe reale a cui appartiene il sample
		lableindex = testlable.max(0)[1].item()

		# sommo alla classe reale l'array che contiene le predizioni
		cmc[lableindex] = cmc[lableindex] + np.asarray(predict.cpu())

		if debug:
			print("a", predict, lableindex)

	# Visualizzo la confusion matrix
	if verbose:
		print(cmc)
	
	# Calcolo accuratezza 
	accuracy = np.sum(cmc.diagonal())/np.sum(cmc)

	# Calcolo precision (true positive / true positive + false positive)
	# recall media rispetto alle classi (true positive / true positive + false negative)
	precision = []
	recall = []
	for i in range(nclasses):
		denominatore = np.sum(cmc[i,:])
		if denominatore > 0:
			recall.append(cmc[i,i] / denominatore)
		else:
			recall.append(1)
		denominatore = np.sum(cmc[:,i])
		if denominatore > 0:
			precision.append(cmc[i,i] / denominatore)
		else:
			precision.append(1)


	precision = np.asarray(precision)
	recall  = np.asarray(recall)

	precision = np.mean(precision)
	recall = np.mean(recall)
	if verbose:
		print('Accuratezza del classificatore: ' + "{0:.2f}".format(accuracy*100) + '%')
		print('Precisione media del classificatore: ' + "{0:.2f}".format(precision))
		print('Recall media del classificatore: ' + "{0:.2f}".format(recall))

	return cmc, accuracy, precision, recall, nfeatures

