import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
# from models import *
# from configs import cfg
import pandas as pd
# from nltk.translate import bleu_score
from utilities import *
from beer_dataloader import *


def process_train_data(texts, beers, ratings, character_only=False):
    """
    Processes a minibatch into one-hot encoding ready for input to the network.
    :param texts:
    :param beers:
    :param ratings:
    :param character_only:
    :return:
    """
    data = texts2oh(texts)
    # if we are not just training the language model.
    if not character_only:
        # concatenate text and metadata.
        metadatas = get_metadatas(beers, ratings)
        data = concat_metadatas(data, metadatas)
    return to_tensor(data)

    
def train_valid_split(data, labels):
    # TODO: Takes in train data and labels as numpy array (or a torch Tensor/ Variable) and
    # splits it into training and validation data.
    raise NotImplementedError
    
    
def process_test_data(data):
    # TODO: Takes in pandas DataFrame and returns a numpy array (or a torch Tensor/ Variable)
    # that has all input features. Note that test data does not contain any review so you don't
    # have to worry about one hot encoding the data.
    raise NotImplementedError


def train(model, X_train, y_train, X_valid, y_valid, cfg):
    # TODO: Train the model!
    raise NotImplementedError
    
    
def generate(model, X_test, cfg):
    # TODO: Given n rows in test data, generate a list of n strings, where each string is the review
    # corresponding to each input row in test data.
    raise NotImplementedError
    
    
def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    raise NotImplementedError


if __name__ == "__main__":
    data_dir = "../BeerAdvocatePA4"
    train_data_fname = data_dir + "/Beeradvocate_Train.csv"
    test_data_fname = data_dir + "/Beeradvocate_Test.csv"
    out_fname = ""

    # train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    # test_data = load_data(test_data_fname) # Generating the pandas DataFrame
    # train_data = utilities.add_limiters(train_data) # for testing.
    # tes = train_data['beer/style'][0]
    # print(train_data.iloc[0])
    # print(utilities.scale_beer_rating(train_data['review/overall'][0]))
    # print(utilities.beer2oh(tes))
    # print(utilities.oh2beer(utilities.beer2oh(tes)))

    train_loader, val_loader = create_split_loaders(2, 42, train_data_fname)
    text1, beers1, rating1 = iter(train_loader).next()
    print(text1, beers1, rating1)
    batch = process_train_data(text1, beers1, rating1, True)
    print(batch)
    # train_data, train_labels = process_train_data(train_data) # Converting DataFrame to numpy array
    # X_train, y_train, X_valid, y_valid = train_valid_split(train_data, train_labels) # Splitting the train data into train-valid data
    # X_test = process_test_data(test_data) # Converting DataFrame to numpy array
    #
    # model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    # if cfg['cuda']:
    #     computing_device = torch.device("cuda")
    # else:
    #     computing_device = torch.device("cpu")
    # model.to(computing_device)
    #
    # train(model, X_train, y_train, X_valid, y_valid, cfg) # Train the model
    # outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    # save_to_file(outputs, out_fname) # Save the generated outputs to a file