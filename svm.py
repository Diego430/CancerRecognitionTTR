from sklearn.svm import SVC
import numpy as np
from utility import loadCancerDatasetPCAShuffle, loadCancerDatasetShuffle

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


verbose = False
debug = False



def createModels(xtrain, ytrain, kernel="linear", maxiteration=10):
	"""
	Creo i models delle SVM necessari 

	Args:
		xtrain: matrice dei sample e features con cui allenare le SVM
		ytrain: array delle label dei campioni 
		kernel: tipologia di kernel da usare per creare i modelli
		maxiteration: numero massimo di iterazioni da usare come limite dalle SVM 

	Returns:
		array di models SVM allenati
	"""
	if verbose:
		print("Building models...")

	# estraggo le classi dalle lable 
	classes = np.unique(ytrain)
	nclasses = classes.shape[0]

	if debug:
		print(f"classes.shape: {nclasses}")
		print(f"classes: {classes}")

	# Inizializzo un modello di classificazione SVM per ognuna delle classi
	models = [SVC(kernel=kernel, max_iter=maxiteration, probability=True) for _ in range(nclasses)]
	
	if verbose:
		print("Training models...")
	# Train dei modelli
	# i modelli seguono l'ordine alfabetico, models[0] modello dell'primo elemento di classes....
	for i in range(nclasses):
		models[i].fit(xtrain, ytrain==classes[i])
	return models



def svmfn(dataset=0, kernelid=0, maxiteration=10, traintest = 0.8, usepca=True):
	"""
	Implemento il riconoscimento usando SVM

	Args:
		dataset: indica il dataset da caricare (0 : brain, 1 : leukemia, 2 : liver)
		kernelid: indica il kernel da usare (0 : linear, 1 : poly, 2 : rbf, 3 : sigmoid)
		maxiteration: numero delle iterazioni massime 
		traintest: percentuale di suddivisione train / test (0.8 = 80% train, 20% test)
		usepca: True per usare database con dimensionalità ridotta dall PCA, False senza PCA
	Returns:
		cmc: confusion matrix
		accuracy: (veri positivi + veri negativi) / totale
		precision: veri positivi / (veri positivi + falsi positivi)
		recall: veri positivi / (veri positivi + falsi negativi) 
		nfeatures: numero delle features usate per il riconoscimento 
	"""

	sampledata = ["cancer/brainSamples.npy", "cancer/leukemiaSamples.npy", "cancer/liverSamples.npy"] 
	labledata = ["cancer/brainLable.npy", "cancer/leukemiaLable.npy", "cancer/liverLable.npy"]
	kernel = ['linear', 'poly', 'rbf', 'sigmoid']
	if verbose:
		print("Loading dataset...")
	
	# Carico dataset
	if usepca:
		xtrain, ytrain, xtest, ytest = loadCancerDatasetPCAShuffle(sampledata[dataset], labledata[dataset], traintest)	
	else:
		xtrain, ytrain, xtest, ytest = loadCancerDatasetShuffle(sampledata[dataset], labledata[dataset], traintest)	
	nfeatures = xtrain.shape[1]

	# classi presenti nel database, sono ordinate in ordine alfabetico 
	classes = np.unique(ytrain)
	nclasses = classes.shape[0]

	if debug:
		print(f"xtrain.shapes: {xtrain.shapes}, ytrain.shape: {ytrain.shape}")
	models = createModels(xtrain, ytrain, kernel[kernelid], maxiteration)

	predicted = []
	if verbose:
		print("Predicting scores...")
	

	# Classifico i dati del testing set 
	# ciclo per sample
	for sample in xtest:
		p = [[] for _ in range(nclasses)]
		# predico se un sample può o meno essere della classe i
		for i in range(nclasses):
			# p contiene predizione per classe, 0 se non appartiene 1 se appartiene 
			p[i] = int(models[i].predict(sample.reshape(1, -1)))
		predicted.append(p)

	predicted = np.asarray(predicted)

	if debug:
		print(f"predicted.shape: {predicted.shape}")
	
	# Creo la confusion matrix
	cmc = np.zeros((nclasses, nclasses))
	for predict, testlable in zip(predicted,ytest):
		# recupero indice della classe reale a cui appartiene il sample
		lableindex = np.where(classes == testlable)[0][0]
		# sommo alla classe reale l'array che contiene le predizioni
		cmc[lableindex] += predict

		if debug:
			print(predict, lableindex)

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
