import numpy as np
import scipy
from scipy.spatial.distance import cdist
from matplotlib import cm
import matplotlib.pyplot as plt
from common.visualization import plot_norm_dist_ellipse as plot_dists

class Lloyd(object):

    def plottIteration(self, samples, codewords, classOfTrainData):
        #plotten
        _, ax = plt.subplots(dpi=150)
        colormap = cm.get_cmap("viridis")(np.linspace(0, 1, len(codewords)))

        for i in range(len(codewords)):
            data = samples[classOfTrainData==i]
            ax.scatter(data[:, 0], data[:, 1], c=colormap[i], edgecolor='k')
        
        plot_dists(ax, codewords, covs, colormap)       

        
#        ax.scatter(samples[:, 0], samples[:, 1], cmap="plasma", edgecolor='k')
#        ax.scatter(codewords[:, 0], codewords[:, 1], cmap="plasma", edgecolor='yellow')

    def cluster(self, samples, codebook_size, prune_codebook=False, verbose=False):
        """Partitioniert Beispieldaten in gegebene Anzahl von Clustern.

        Params:
            samples: ndarray mit Beispieldaten, shape=(d,t)
                mit d Beispieldaten und t dimensionalen Merkmalsvektoren.
            codebook_size: Anzahl von Komponenten im Codebuch.
            prune_codebook: Boolsches Flag, welches angibt, ob das Codebuch
                bereinigt werden soll. Die Bereinigung erfolgt auf Grundlage
                einer Heuristik, die die Anzahl der, der Cluster Komponente
                zugewiesenen, Beispieldaten beruecksichtigt.
                Optional, default=False

        Returns:
            codebook: ndarry mit codebook_size Codebuch Vektoren,
                zeilenweise, shape=(codebook_size,t)
        """

        resampled_train_data = np.random.permutation(samples)
        codewords = resampled_train_data[:codebook_size]

        lastError = 1000
        relativeError = 1000
        threshhold_for_relative_error = 0.005



        while (relativeError > threshhold_for_relative_error):
            #distanzen und neuzuweisung der trainingsdaten zu cluster
            distancesBetweenTrainAndTest = scipy.spatial.distance.cdist(samples, codewords, metric='euclidean')

            classOfTrainData = np.zeros(distancesBetweenTrainAndTest.shape[0], dtype=int)
            closestClusterDistance = np.zeros(distancesBetweenTrainAndTest.shape[0])

            for i in range(distancesBetweenTrainAndTest.shape[0]):
                closestCluster = distancesBetweenTrainAndTest[i].argmin()
                closestClusterDistance[i] = distancesBetweenTrainAndTest[i].min()
                classOfTrainData[i] = closestCluster

            #quantisierungsfehler 
            error = np.mean(closestClusterDistance)

            #neu center of centroiden
            for i in range(len(codewords)):
                newCordWord = np.zeros(2, dtype=float)

                newCordWord[0] = np.mean(samples[classOfTrainData==i][:,0])
                newCordWord[1] = np.mean(samples[classOfTrainData==i][:,1])
                
                codewords[i] = newCordWord

            relativeError = np.abs(lastError - error)
            lastError = error
            if(verbose):
                print(f"New Error: {error} | relative Error {relativeError}")
        
        if(verbose):
            self.plottIteration(samples, codewords, classOfTrainData)

        return (codewords,classOfTrainData)


        #
        # Bestimmen Sie in jeder Iteration den Quantisierungsfehler und brechen Sie
        # das iterative Verfahren ab, wenn der Quantisierungsfehler konvergiert
        # (die Aenderung einen sehr kleinen Schwellwert unterschreitet).
        # Nuetzliche Funktionen: scipy.distance.cdist, np.mean, np.argsort
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        # Fuer die Initialisierung mit zufaelligen Punkten: np.random.permutation
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.permutation.html
        