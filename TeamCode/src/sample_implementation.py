import numpy as np
from sklearn.ensemble import RandomForestClassifier
from TeamCode.src.interface import AbstractDigitizationModel, AbstractClassificationModel

import helper_code as hc
import os
import pickle

import torch
from tqdm import tqdm
# import cv2
from helper_code import load_images

from sklearn.cluster import KMeans

class ExampleDigitizationModel(AbstractDigitizationModel):
    def __init__(self):
        pass

    @classmethod
    def from_folder(cls, model_folder, verbose):
        filename = os.path.join(model_folder, 'digitization_model.pkl')
        model_instance = load_pickle(filename)
        return model_instance

    def train_model(self, data_folder, model_folder, verbose):
        if verbose:
            print('Training the digitization model...')
            print('Finding the Challenge data...')

        records = hc.find_records(data_folder)
        num_records = len(records)

        if num_records == 0:
            raise FileNotFoundError('No data was provided.')

        # Extract the features and labels.
        if verbose:
            print('Extracting features and labels from the data...')

        features = list()

        for i in range(num_records):
            if verbose:
                width = len(str(num_records))
                print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

            record = os.path.join(data_folder, records[i])

            # Extract the features from the image...
            current_features = extract_features(record)
            features.append(current_features)

        # Train the model.
        if verbose:
            print('Training the model on the data...')

        # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
        self.model_random_seed = np.mean(features)

        # Create a folder for the model if it does not already exist.
        os.makedirs(model_folder, exist_ok=True)
        full_path = os.path.join(model_folder, 'digitization_model.pkl')
        save_pickle(self, full_path)

        if verbose:
            print('Done.')
            print()

    
    def run_digitization_model(self, record, verbose):
        model_seed = self.model_random_seed

        # Extract features.
        features = extract_features(record)

        # Load the dimensions of the signal.
        header_file = hc.get_header_file(record)
        header = hc.load_text(header_file)

        num_samples = hc.get_num_samples(header) # oversample and see if the evaluation can handle it
        num_signals = hc.get_num_signals(header)

        # For a overly simply minimal working example, generate "random" waveforms.
        seed = int(round(model_seed + np.mean(features)))
        signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
        signal = np.asarray(signal, dtype=np.int16)
        return signal


class ExampleClassificationModel(AbstractClassificationModel):

    @classmethod
    def from_folder(cls, model_folder, verbose):
        filename = os.path.join(model_folder, 'dx_model.pkl')
        model_instance = load_pickle(filename)
        return model_instance


    # Train your dx classification model.
    def train_model(self, data_folder, model_folder, verbose):
        # Find data files.
        if verbose:
            print('Training the dx classification model...')
            print('Finding the Challenge data...')

        records = hc.find_records(data_folder)
        num_records = len(records)

        if num_records == 0:
            raise FileNotFoundError('No data was provided.')

        # Extract the features and labels.
        if verbose:
            print('Extracting features and labels from the data...')

        features = list()
        dxs = list()

        for i in range(num_records):
            if verbose:
                width = len(str(num_records))
                print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

            record = os.path.join(data_folder, records[i])

            # Extract the features from the image, but only if the image has one or more dx classes.
            dx = hc.load_dx(record)
            if dx:
                current_features = extract_features(record)
                features.append(current_features)
                dxs.append(dx)

        if not dxs:
            raise Exception('There are no labels for the data.')

        features = np.vstack(features)
        self.classes = sorted(set.union(*map(set, dxs)))
        dxs = hc.compute_one_hot_encoding(dxs, self.classes)

        # Train the model.
        if verbose:
            print('Training the model on the data...')

        # Define parameters for random forest classifier and regressor.
        n_estimators   = 12  # Number of trees in the forest.
        max_leaf_nodes = 34  # Maximum number of leaf nodes in each tree.
        random_state   = 56  # Random state; set for reproducibility.

        # Fit the model.
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, dxs)

        # Create a folder for the model if it does not already exist.
        os.makedirs(model_folder, exist_ok=True)

        # Save the model.
        path = os.path.join(model_folder, 'dx_model.pkl')
        save_pickle(self, path)

        if verbose:
            print('Done.')
            print()

    def run_classification_model(self, record, signal, verbose):
        model = self.model
        classes = self.classes

        # Extract features.
        features = extract_features(record)
        features = features.reshape(1, -1)

        # Get model probabilities.
        probabilities = model.predict_proba(features)
        probabilities = np.asarray(probabilities, dtype=np.float32)[:, 0, 1]

        # Choose the class(es) with the highest probability as the label(s).
        max_probability = np.nanmax(probabilities)
        labels = [classes[i] for i, probability in enumerate(probabilities) if probability == max_probability]

        return labels


# # Save your trained digitization model.
# def save_digitization_model(model_folder, model):
#     d = {'model': model}
#     filename = os.path.join(model_folder, 'digitization_model.sav')
#     joblib.dump(d, filename, protocol=0)

# # Save your trained dx classification model.
# def save_dx_model(model_folder, model, classes):
#     d = {'model': model, 'classes': classes}
#     filename = os.path.join(model_folder, 'dx_model.sav')
#     joblib.dump(d, filename, protocol=0)

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):  
    with open(path, 'rb') as f:
        return pickle.load(f)



################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    images = hc.load_images(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])

