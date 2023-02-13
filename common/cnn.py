import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        """Initialisierung eines CNNs zur Klassifikation des MNIST Datensatzes.

        Initialisierung der Convolutional und der Fully Connected Schichten
        """
        raise NotImplementedError()


    def forward(self, x):
        """Vorwaertsdurchlauf durch das CNN.

        Params:
            x: tensor, Minibatch, fuer die die Ausgabe berechnet wird.
        """
        raise NotImplementedError()



    def test(self, test_loader):
        """Evaluiert das Netz indem Loss, Accuracy und die Anzahl korrekt
        klassifiezierter Beispiele fuer den Testdatensatz berechnet werden.

        Params:
            test_loader: torch dataloader fuer den Testdatensatz
        """
        self.eval()

        raise NotImplementedError()


        return test_loss, accuracy, correct
