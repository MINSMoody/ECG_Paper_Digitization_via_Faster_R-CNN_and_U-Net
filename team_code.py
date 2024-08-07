#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import numpy as np
import os
from TeamCode.src.implementation import OurDigitizationModel, VoidClassificationModel
from TeamCode.src.verify_environment import verify_environment
import sys

from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your digitization model.
def train_models(data_folder, model_folder, verbose):
    digitization_model = OurDigitizationModel()
    classification_model = VoidClassificationModel()

    digitization_model.train_model(data_folder, model_folder, verbose)
    classification_model.train_model(data_folder, model_folder, verbose)


# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
# outputs tuple (digitization model, classification model)
def load_models(model_folder, verbose):
    digitization_model = OurDigitizationModel.from_folder(model_folder, verbose)
    classification_model = VoidClassificationModel.from_folder(model_folder, verbose)

    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):

    # Run the digitization model; if you did not train this model, then you can set signal = None.
    signal = digitization_model.run_digitization_model(record, verbose)

    # Run the classification model; if you did not train this model, then you can set labels = None.
    labels = classification_model.run_classification_model(record, signal, verbose)

    return signal, labels
