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


def load_data(fname):
    # TODO: From the csv file given by filename and return a pandas DataFrame of the read csv.
    data = pd.read_csv(open(fname, 'r'))
    to_drop = ['Unnamed: 0', 'beer/name', 'beer/beerId', 'beer/brewerId', 'beer/ABV',
               'review/appearance', 'review/aroma', 'review/palate', 'review/taste',
               'review/time', 'review/profileName']
    data = data.drop(columns=to_drop)
    return data


def process_train_data(data):
    # TODO: Input is a pandas DataFrame and return a numpy array (or a torch Tensor/ Variable)
    # that has all features (including characters in one hot encoded form).
    raise NotImplementedError

    
def train_valid_split(data, labels):
    # TODO: Takes in train data and labels as numpy array (or a torch Tensor/ Variable) and
    # splits it into training and validation data.
    raise NotImplementedError
    
    
def process_test_data(data):
    # TODO: Takes in pandas DataFrame and returns a numpy array (or a torch Tensor/ Variable)
    # that has all input features. Note that test data does not contain any review so you don't
    # have to worry about one hot encoding the data.
    raise NotImplementedError

    
def pad_data(orig_data):
    # TODO: Since you will be training in batches and training sample of each batch may have reviews
    # of varying lengths, you will need to pad your data so that all samples have reviews of length
    # equal to the longest review in a batch. You will pad all the sequences with <EOS> character 
    # representation in one hot encoding.
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


def add_limiters(data):
    """
    Adds limiters and strips tab characters from review/text. To be run on a Pandas Dataframe
    before other processing.
    :param data: Pandas Dataframe.
    :return: Pandas Dataframe.
    """
    temp = data.copy()
    temp['review/text'] = temp['review/text'].str.replace('\t', ' ') # TODO is this necessary?
    temp['review/text'] = '^' + temp['review/text'].str.strip() + '`' # start with ascii 94 end 96.
    return temp


def char2oh(text):
    """
    Converts a string to a one-hot encoded 2D Tensor.
    :param text: String of the review to be encoded.
    :return: 2D Tensor. One hot encoded.
    """
    values = []
    # get ascii representation, create tensor and add one to index for each character
    for char in text:
        index = ord(char) - 32  # we don't use 0-31 as non printable.
        temp = torch.zeros((1, 98), dtype=torch.float)
        temp[0][index] = 1.0
        values.append(temp)
    return torch.stack(values)

def oh2char(tensor):
    """
    Converts one-hot encoded values to a string.
    :param tensor: 2D Tensor of one-hot encoded values
    :return: String.
    """
    chars = []
    # iterate through tensor, get characters and add to list.
    for row in tensor:
        num = torch.argmax(row).item() + 32  # correct for our shifted index.
        chars.append(chr(num))
    return ''.join(chars)


if __name__ == "__main__":
    data_dir = "../BeerAdvocatePA4"
    train_data_fname = data_dir + "/Beeradvocate_Train.csv"
    test_data_fname = data_dir + "/Beeradvocate_Test.csv"
    out_fname = ""

    train_data = load_data(train_data_fname) # Generating the pandas DataFrame
    test_data = load_data(test_data_fname) # Generating the pandas DataFrame
    train_data = add_limiters(train_data) # for testing.
    tes = char2oh(train_data['review/text'][0])
    print(tes)
    print(oh2char(tes))

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