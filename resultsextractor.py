import csv
import numpy as np

filenames = ['results_svm.csv', 'results_kmeans.csv', 'results_nn.csv']
filename = filenames[0]
with open(filename, 'r') as file:
	reader = csv.reader(file)
	header = True
	cancer = []
	kernel = []
	pca = []
	filelist = []
	for row in reader:
		if header:
			header = False
			continue
		else:
			cancer.append(row[0])
			kernel.append(row[1])
			pca.append(row[2])
			filelist.append(row)

print(cancerclass)

