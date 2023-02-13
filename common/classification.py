from collections import defaultdict
import numpy as np
import scipy.spatial.distance
from scipy.spatial.distance import cdist
from common import log_math
from collections import Counter


class KNNClassifier(object):

    def __init__(self, k_neighbors, metric):
        """Initialisiert den Klassifikator mit Meta-Parametern

        Params:
            k_neighbors: Anzahl der zu betrachtenden naechsten Nachbarn (int)
            metric: Zu verwendendes Distanzmass (string),
                siehe auch scipy Funktion cdist
        """
        self.k_neighbors = k_neighbors
        self.metric = metric
        #raise NotImplementedError()


    def estimate(self, train_samples, train_labels):
        """Erstellt den k-Naechste-Nachbarn Klassfikator mittels Trainingdaten.

        Der Begriff des Trainings ist beim K-NN etwas irre fuehrend, da ja keine
        Modelparameter im eigentlichen Sinne statistisch geschaetzt werden.
        Diskutieren Sie, was den K-NN stattdessen definiert.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """

        self.train_samples = train_samples
        self.train_labels = train_labels
        #raise NotImplementedError()


    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Implementieren Sie die Klassifikation der Daten durch den KNN.
        #
        # Nuetzliche Funktionen: scipy.spatial.distance.cdist, np.argsort
        # http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
        
        distancesBetweenTrainAndTest = scipy.spatial.distance.cdist(test_samples, self.train_samples, metric=self.metric)

        calc_test_labels = np.zeros(len(distancesBetweenTrainAndTest),dtype=np.str_)
        k = self.k_neighbors

        def most_commen(array):
            #return max(set(array), key=array.count)
            data = Counter(array)
            #print(data)
            return data.most_common(1)[0][0]
        
        for i in range(distancesBetweenTrainAndTest.shape[0]):
            if k<1:
                print("Error") 
            elif k==1:
                nearestNeigbor = distancesBetweenTrainAndTest[i].argmin()
                calc_test_labels[i] = self.train_labels[nearestNeigbor]
            else: 
                nearestNeighbors = np.argsort(distancesBetweenTrainAndTest[i])[:k]
                nearestNeighbors_labels = self.train_labels[nearestNeighbors]
                calc_test_labels[i] = most_commen(nearestNeighbors_labels)

        return calc_test_labels       
        #raise NotImplementedError()



class GaussianClassifier(object):

    def __init__(self):
        self.classProp = []
        self.normalDistributions = []
        self.class_obj_labels = 0
        """Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.
        """


    def estimate(self, train_samples, train_labels):
        """Erstellt den Normalverteilungsklassikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        from common.data_provider import DataProvider
        
        train_data_provider = DataProvider(DataProvider.DATA2DROOT_TRAIN)
        number_of_Samples = len(train_samples)
        labels = np.unique(train_labels)
        self.class_obj_labels = labels

        mean_list = []
        cov_list = []
        for label in labels:
            class_data = train_data_provider.get_class_arr(class_idx=label)
            #print(class_data)
            # Berechnen Sie mean und cov hier
            class_data_len = len(class_data)
            mean = np.sum(class_data, axis=0) / class_data_len
            cov = np.cov(class_data, rowvar=0)
            np.testing.assert_almost_equal(actual=mean,
                                        desired=np.mean(class_data, axis=0),
                                        err_msg='Der Mittelwert ist falsch')
            np.testing.assert_almost_equal(actual=cov,
                                        desired=np.cov(class_data, rowvar=0),
                                        err_msg='Die Kovarianzmatrix ist falsch')
            mean_list.append(mean)
            cov_list.append(cov)

        for label in labels:
            number_of_samples_with_class = len(class_data)
            p_class = number_of_samples_with_class / number_of_Samples
            self.classProp.append(p_class)

        print(mean_list)
        for i in range(len(labels)):
            normal_disti = scipy.stats.norm(mean_list[i], cov_list[i])
            self.normalDistributions.append(normal_disti)



    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Werten Sie die Dichten aus und klassifizieren Sie die
        # Testdaten.
        #
        # Hinweise:
        #
        # Durch welche geeignete monotone Transformation lassen sich numerische
        # Probleme bei der Auswertung von extrem kleinen Dichtewerten vermeiden?
        # Beruecksichtigen Sie das in Ihrer Implementierung.
        #
        # Erstellen Sie fuer die Auswertung der transformierten Normalverteilung
        # eine eigene Funktion. Diese wird in den folgenden Aufgaben noch von
        # Nutzen sein.

        testLabels = []

        for s in range(len(test_samples)):
            sample = test_samples[s]
            propies = np.zeros(len(self.class_obj_labels))

            for i in range(len(self.class_obj_labels)):
                dist_prop = self.normalDistributions[i].pdf(sample)[0][0]
                class_prop = self.classProp[i]
                propies[i] = class_prop * dist_prop
                            
            testLabels.append(self.class_obj_labels[propies.argmax()])
            #print(f"Iteration: {s} | Wht: {max(propies)} | Label: {testLabels[s]}")

        return testLabels
        #raise NotImplementedError()


class MDClassifierClassIndep(object):

    def __init__(self, quantizer, num_densities):
        """Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.

        Params:
            quantizer: Objekt, das die Methode cluster(samples,codebook_size,prune_codebook)
                implementiert. Siehe Klasse common.vector_quantization.Lloyd
            num_densities: Anzahl von Mischverteilungskomponenten, die verwendet
                werden sollen.
        """
        raise NotImplementedError()


    def estimate(self, train_samples, train_labels):
        """Erstellt den Mischverteilungsklassifikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Diese Methode soll das Training eines Mischverteilungsklassifikators
        # mit klassenunabhaengigen Komponentendichten implementieren (siehe Skript S. 67 f.).
        #
        # Die folgenden Funtionen koennen bei der Implementierung von Nutzen
        # sein:
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.exp.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.inv.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.log.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.slogdet.html
        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.bincount.html

        #
        # Schaetzen Sie das GMM der Trainingsdaten.
        #
        # Wieviele Datenpunkte werden zur Schaetzung jeder Normalverteilung mindestens
        # benoetigt und welche Eigenschaften muessen diese haben?
        # Beruecksichtigen Sie das in Ihrer Implementierung.

        raise NotImplementedError()


        #
        # Bestimmen Sie fuer jede Klasse die spezifischen Mischungsgewichte.
        # Beachten Sie, dass die Dichteauswertung wieder ueber eine geeignete
        # monotome Transformationsfunktion geschehen soll. Verwenden Sie hierfuer
        # die Funktion, die Sie bereits fuer den GaussianClassifier implementiert
        #
        # Achten Sie darauf, dass sich die Mischungsgewichte zu 1 addieren.

        raise NotImplementedError()


    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError()


class MDClassifierClassDep(object):

    def __init__(self, quantizer, num_densities):
        """Initialisiert den Klassifikator
        Legt Klassenvariablen fuer die Modellparameter an.

        Params:
            quantizer: Objekt, das die Methode cluster(samples,codebook_size,prune_codebook)
                implementiert. Siehe Klasse common.vector_quantization.Lloyd
            num_densities: Anzahl von Mischverteilungskomponenten, die je Klasse
                verwendet werden sollen.
        """
        raise NotImplementedError()


    def estimate(self, train_samples, train_labels):
        """Erstellt den Mischverteilungsklassifikator mittels Trainingdaten.

        Schaetzt die Modellparameter auf Grundlage der Trainingsdaten.

        Hinweis: Die Funktion heisst estimate im Sinne von durchgaengigen Interfaces
                fuer das Duck-Typing

        Params:
            train_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.
        """
        #
        # Schaetzen Sie die jeweils ein GMM fuer jede Klasse.
        #
        # Tipp: Verwenden Sie die bereits implementierte Klasse MDClassifierClassIndep

        raise NotImplementedError()


    def classify(self, test_samples):
        """Klassifiziert Test Daten.

        Params:
            test_samples: ndarray, das Merkmalsvektoren zeilenweise enthaelt (d x t).

        Returns:
            test_labels: ndarray (1-Dimensional), das Klassenlabels enthaelt
                (d Komponenten, train_labels.shape=(d,) ).

        mit d Testbeispielen und t dimensionalen Merkmalsvektoren.
        """
        raise NotImplementedError()


class NeuralNetwork(object):

    def __init__(self, learning_rate, iterations, activation='Linear'):
        """Initialisiert das Perzeptron und die Klassen Variablen als Modellparameter.

        Params:
            learning_rate: Faktor zur Gewichtung der Gewichtsupdates.
            iterations: Anzahl der Trainingsiterationen.
            activation: String, definiert die Aktivierungsfunktion des Ausgabeneurons
                'Linear', 'TanH', 'Sigmoid', ...
        """
        self.__learning_rate = learning_rate
        self.__iterations = iterations
        # Speichern Sie den Root-Mean-Squared Error und die Gewichte fuer jede Trainingsiteration
        self.__nn_rms_list = []
        self.__nn_weights_list = []

        # Initialisieren Sie die Aktivierungsfunktionen und deren Ableitungen mithilfe des lambda Operators
        # https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions
        # if activation == 'Linear':
        #    self.__activation = lambda x: ...
        #    self.__activation_deriv = lambda x: ...
        raise NotImplementedError()


    def estimate(self, train_samples, train_labels):
        """Trainiert das neuronale Netzwerk mit den gegebenen Trainingsdaten.

        Optimiert die Modellparameter basierend auf den Trainingsdaten
        mit d Trainingsbeispielen und t dimensionalen Merkmalsvektoren.

        Params:
            train_samples: ndarray, welches die Merkmalsvektorenthat zeilenweise enthaelt (d x t).
            train_labels: ndarray (1-dimensional), Klassenlabels
                (d components, train_labels.shape=(d,) ).
        """
        raise NotImplementedError()



    def classify(self, test_samples, iteration=None):
        """Klassifiziert die Testdaten
        Params:
            test_samples: ndarray mit Testdaten (Nxdim) mit N der Anzahl Daten und
               dim der Anzahl Dimensionen pro Sample
            iteration: Fuehrt die Klassifikation mit Gewichten einer bestimmten Iteration
               durch. (Nuetzlich fuer die Visualisierung). Default: None, nehme die Gewichte
               der letzten Iteration.
        """
        raise NotImplementedError()


    @property
    def nn_rms_list(self):
        raise NotImplementedError()


    def activation(self, x_input):
        """Berechnet Aktivierungsfunktion."""
        # Wird fuer unittest der Aktivierungsfunktion benoetigt
        return self.__activation(x_input)

    def activation_deriv(self, x_input):
        """Berechnet die Ableitung der Aktivierungsfunktion"""
        # Wird fuer unittest der Aktivierungsfunktion benoetigt
        return self.__activation_deriv(x_input)

class CrossValidation(object):

    def __init__(self, samples, labels, n_folds):
        """Initialisiert die Kreuzvalidierung

        Params:
            samples: ndarray mit Beispieldaten, shape=(d,t)
            labels: ndarray mit Labels fuer Beispieldaten, shape=(d,)
            n_folds: Anzahl Folds ueber die die Kreuzvalidierung durchgefuehrt
                werden soll.

        mit d Beispieldaten und t dimensionalen Merkmalsvektoren.
        """
        self.__samples = samples
        self.__labels = labels
        self.__n_folds = n_folds

    def validate(self, classifier):
        """Fuert die Kreuzvalidierung mit dem Klassifikator 'classifier' durch.

        Params:
            classifier: Objekt, das folgende Methoden implementiert (siehe oben)
                estimate(train_samples, train_labels)
                classify(test_samples) --> test_labels

        Returns:
            crossval_overall_result: Erkennungsergebnis der gesamten Kreuzvalidierung
                (ueber alle Folds)
            crossval_class_results: Liste von Tuple (category, result) die klassenweise
                Erkennungsergebnisse der Kreuzvalidierung enthaelt.
        """
        crossval_overall_list = []
        crossval_class_dict = defaultdict(list)
        for fold_index in range(self.__n_folds):
            train_samples, train_labels, test_samples, test_labels = self.samples_fold(fold_index)
            classifier.estimate(train_samples, train_labels)
            estimated_test_labels = classifier.classify(test_samples)
            classifier_eval = ClassificationEvaluator(estimated_test_labels, test_labels)
            crossval_overall_list.append(list(classifier_eval.error_rate()))
            crossval_class_list = classifier_eval.category_error_rates()
            for category, err, n_wrong, n_samples in crossval_class_list:
                crossval_class_dict[category].append([err, n_wrong, n_samples])

        crossval_overall_mat = np.array(crossval_overall_list)
        crossval_overall_result = CrossValidation.__crossval_results(crossval_overall_mat)

        crossval_class_results = []
        for category in sorted(crossval_class_dict.keys()):
            crossval_class_mat = np.array(crossval_class_dict[category])
            crossval_class_result = CrossValidation.__crossval_results(crossval_class_mat)
            crossval_class_results.append((category, crossval_class_result))

        return crossval_overall_result, crossval_class_results

    @staticmethod
    def __crossval_results(crossval_mat):
        # Relative number of samples
        crossval_weights = crossval_mat[:, 2] / crossval_mat[:, 2].sum()
        # Weighted sum over recognition rates for all folds
        crossval_result = (crossval_mat[:, 0] * crossval_weights).sum()
        return crossval_result

    def samples_fold(self, fold_index):
        """Berechnet eine Aufteilung der Daten in Training und Test

        Params:
            fold_index: Index des Ausschnitts der als Testdatensatz verwendet werden soll.

        Returns:
            train_samples: ndarray mit Trainingsdaten, shape=(d_train,t)
            train_label: ndarray mit Trainingslabels, shape=(d_train,t)
            test_samples: ndarray mit Testdaten, shape=(d_test,t)
            test_label: ndarray mit Testlabels, shape=(d_test,t)

        mit d_{train,test} Beispieldaten und t dimensionalen Merkmalsvektoren.
        """
        n_samples = self.__samples.shape[0]
        test_indices = range(fold_index, n_samples, self.__n_folds)
        train_indices = [train_index for train_index in range(n_samples)
                             if train_index not in test_indices]

        test_samples = self.__samples[test_indices, :]
        test_labels = self.__labels[test_indices]
        train_samples = self.__samples[train_indices, :]
        train_labels = self.__labels[train_indices]

        return train_samples, train_labels, test_samples, test_labels

class ClassificationEvaluator(object):

    def __init__(self, estimated_labels, groundtruth_labels):
        """Initialisiert den Evaluator fuer ein Klassifikationsergebnis
        auf Testdaten.

        Params:
            estimated_labels: ndarray (1-Dimensional) mit durch den Klassifikator
                bestimmten Labels (N Komponenten).
            groundtruth_labels: ndarray (1-Dimensional) mit den tatsaechlichen
                Labels (N Komponenten).
        """
        self.__estimated_labels = estimated_labels
        self.__groundtruth_labels = groundtruth_labels
        self.__binary_result_mat = groundtruth_labels == estimated_labels

    def error_rate(self, mask=None):
        """Bestimmt die Fehlerrate auf den Testdaten.

        Params:
            mask: Optionale boolsche Maske, mit der eine Untermenge der Testdaten
                ausgewertet werden kann. Nuetzlich fuer klassenspezifische Fehlerraten.
                Bei mask=None werden alle Testdaten ausgewertet.
        Returns:
            tuple: (error_rate, n_wrong, n_samlpes)
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        if mask is None:
            mask = np.ones_like(self.__binary_result_mat, dtype=bool)
        masked_binary_result_mat = self.__binary_result_mat[mask]
        n_samples = len(masked_binary_result_mat)
        n_correct = masked_binary_result_mat.sum()
        n_wrong = n_samples - n_correct
        error_rate = n_wrong / float(n_samples)
        error_rate *= 100
        return error_rate, n_wrong, n_samples

    def category_error_rates(self):
        """Berechnet klassenspezifische Fehlerraten

        Returns:
            list von tuple: [ (category, error_rate, n_wrong, n_samlpes), ...]
            category: Label der Kategorie / Klasse
            error_rate: Fehlerrate in Prozent
            n_wrong: Anzahl falsch klassifizierter Testbeispiele
            n_samples: Gesamtzahl von Testbeispielen
        """
        category_list = sorted(set(self.__groundtruth_labels.ravel()))
        cat_n_err_list = []
        for category in category_list:
            category_mask = self.__groundtruth_labels == category
            err, n_wrong, n_samples = self.error_rate(category_mask)
            cat_n_err_list.append((category, err, n_wrong, n_samples))

        return cat_n_err_list

    def print_results(self):
        """Gibt die Ergebnisse der Evaluation in der Konsole aus"""
        print('Fehlerrate: %.1f; Anzahl falsch-klassifizierte Muster: %3d; Anzahl Muster: %3d' % self.error_rate())
        print('Klassenspezifische Fehlerraten')
        for res in self.category_error_rates():
            print('Klasse %s:\tFehlerrate: %.1f;\tAnzahl falsch-klassifizierte Muster: %3d;\tAnzahl Muster: %3d' % res)

