import itertools

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


class GRL_RATE(keras.callbacks.Callback):
    def __init__(self,number_of_epochs=200):
        super(GRL_RATE, self).__init__()
        self.number_of_epochs = number_of_epochs


    def on_epoch_begin(self, epoch, logs={}):
        p = np.float(epoch) / self.number_of_epochs
        grl = 2. / (1. + np.exp(-10. * p)) - 1
        self.model.layers[-8].hp_lambda = grl
        print(
            f'The current epoch is {epoch}.'
        )
        print(
            f'The current grl is {grl}.'
        )
