import numpy as np
from sklearn.decomposition import PCA


verbose = False
debug = False

def raw_pca(data,nfeatures):
	"""
	Funzione che implementa la Principal Component Analysis - PCA
	
	Args:
		data: matrice delle features da ridurre

	Returns:
		matrice ridotta
	"""
	
	# Calcolo la media di ogni feature e centro i dati
	# data = data.transpose()
	media = np.mean(data, axis=0)
	datac = (data - media).transpose()
	if verbose:
		print(datac.shape)

	# Calcolo la matrice di covarianza dei dati centrati
	mcovarianza = np.cov(datac, rowvar=False)  
	if verbose:
		print("cov Done")

	# Calcolo autovalori e autovettori della matrice di covarianza

	autovalori, autovettori = np.linalg.eigh(mcovarianza)
	if verbose:
		print("autovettori autovalori Done")

	# Ordino gli autovalori dal più grande al più piccolo
	indexBestAutovalori = np.argsort(autovalori)[::-1]
	bestAutovalori = autovalori[indexBestAutovalori]
	bestAutovettori = autovettori[:,indexBestAutovalori]

	if verbose:
		print("bestAutovettori bestAutovalori Done")

	if debug:
		print(bestAutovettori.shape)
	# Costruisco la matrice di trasformazione T: 
	# - in colonna gli autovettori corrispondenti ai nfeatures autovalori più grandi
	Tmatrix = bestAutovettori[:,:nfeatures]

	# Applico la trasformazione 
	Tdata = np.dot(datac, Tmatrix).transpose()
	if verbose:
		print("dot Done")
	if debug:
		print(Tdata.shape)

	return Tdata

def pcafn(samples, nfeatures):
	"""
	Funzione che implementa la Principal Component Analysis - PCA utilizzando la classe di sklearn
	
	Args:
		samples: matrice delle features da ridurre

	Returns:
		matrice ridotta
	"""
	# istanzio classe PCA
	pca = PCA(n_components=nfeatures)

	# alleno PCA con campione ed estraggo risultato PCA
	results = pca.fit_transform(samples)


	if debug:
		print(f"pca extraction result.shape: {results.shape}")
	
	return results

