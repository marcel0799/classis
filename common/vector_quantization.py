import numpy as np
from scipy.spatial.distance import cdist

class Lloyd(object):

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

        raise NotImplementedError()


