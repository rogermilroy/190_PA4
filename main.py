import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
from models import *
from configs import cfg
from utilities import *
from beer_dataloader import *
import time


def process_train_data(texts, beers, ratings, character_only=False):
    """
    Processes a minibatch into one-hot encoding ready for input to the network.
    :param texts: A minibatch of reviews as a tuple.
    :param beers: A minibatch of beers as a tuple
    :param ratings: A minibatch of ratings as a tuple.
    :param character_only: True if we are only training the language model.
    :return: tensor dims (s x N x d) N is minibatch size, s is sequence length and d is 98 if
    char_only or 203 otherwise.
    """
    data = texts2oh(texts)
    # if we are not just training the language model.
    if not character_only:
        # concatenate text and metadata.
        metadatas = get_metadatas(beers, ratings)
        data = concat_metadatas(data, metadatas)
    return to_tensor(data)


def process_test_data(beers, ratings):
    """
    Processesd a minibatch of test data into one hot encoding.
    :param beers: A minibatch of beers as a tuple.
    :param ratings: A minibatch of ratings as a tuple.
    :return: Tensor dims (N x d)
    """
    data = get_metadatas(beers, ratings)
    return torch.stack(data)


def train(model, train_loader, val_loader, cfg):
    # TODO: Train the model!

    num_epochs = cfg['epochs']
    print_every = 1
    plot_every = 2000
    learning_rate = cfg['learning_rate']
    in_size = 1 # TODO CHANGE TO CORRECT DIMENSION
    val_size = 1 # TODO CHANGE TO CORRECT DIMENSION

    # use adam optimizer with default params and given learning rate
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # use cross entropy loss as loss function
    criterion = nn.CrossEntropyLoss()

    # save important stats
    start_time = time.time()
    all_losses = []
    training_loss_avg = 0

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        # Get next minibatch of data for training
        torch.cuda.empty_cache()
        for minibatch_count, (text, beer, rating) in enumerate(train_loader, 0):

            batch = process_train_data(text, beer, rating, True)

            # training
            model.zero_grad()
            training_loss = 0
            for c in range(len(text)):
                tens = torch.unsqueeze(batch[c], 0)
                output = model(tens)
                targets = to_indices(batch[c+1])
                crit_inputs = torch.squeeze(output)
                training_loss += criterion(crit_inputs, targets)

            training_loss.backward(retain_graph=True)
            optimizer.step()

            # calculate loss
            training_loss = training_loss / len(text)
            training_loss_avg += training_loss

            # Calculate validation of every plot_every minibatches
            if minibatch_count % plot_every == 0:

                # Get next minibatch of data for validation
                torch.cuda.empty_cache()
                for minibatch_count, (text, beer, rating) in enumerate(val_loader, 0):

                    batch = process_train_data(text, beer, rating, True)

                    # validation
                    validation_loss = 0
                    for c in range(len(text)):
                        tens = torch.unsqueeze(batch[c], 0)
                        output = model(tens)
                        targets = to_indices(batch[c+1])
                        crit_inputs = torch.squeeze(output)
                        validation_loss += criterion(crit_inputs, targets)

                    # calculate loss
                    validation_loss = validation_loss / val_size
                    # break if loss goes up too many times consecutively
                    if(False):
                        # TODO BREAK AFTER VALIDATION LOSS INCREASES
                        break;


        # plotting and printing every n epochs
        if epoch % print_every == 0:
            print('[%s] (epoch: %d - %d%%)' % (time_since(start), epoch, epoch / num_epochs * 100))
            print('Training Loss: %d' % training_loss)
            print('Validation Loss: %d' % validation_loss)

            print('Generated Text: ', generate(model, None,cfg))

        if epoch % plot_every == 0:
            all_losses.append(loss_avg / plot_every)
            loss_avg = 0




def generate(model, batch, cfg):
    """
    Given n rows in test data, generate a list of n strings, where each string is the review
    corresponding to each input row in test data.
    :param model:
    :param batch:
    :param cfg:
    :return:
    """
    # Initialise a list of SOS characters. TODO test!!
    letters = [char2oh('^') for i in range(len(batch))]
    gen_texts = []

    # Loop until only EOS is predicted.
    while not all_finished(letters):
        # format the data for input to the network.
        inp = concat_metadata(letters, batch)
        outputs = model.forward(inp)
        # sample from softmax distribution.
        letters = get_predicted_letters(outputs)
        gen_texts.append(letters)
    # convert to strings and return.
    return oh2texts(gen_texts)

def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    with open(fname, 'w') as file:
        for output in outputs:
            file.write(output + '\n')  # TODO test!


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
    print("Text: ", text1, "Beers: ", beers1, "Rating: ", rating1)
    batch = process_train_data(text1, beers1, rating1, True)
    print("Batch: ", batch)
    print("Batch Dimensions: ", batch.shape)
    # train_data, train_labels = process_train_data(train_data) # Converting DataFrame to numpy array
    # X_train, y_train, X_valid, y_valid = train_valid_split(train_data, train_labels) # Splitting the train data into train-valid data
    # X_test = process_test_data(test_data) # Converting DataFrame to numpy array
    #
    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)

    generate(model, batch, cfg)

    train(model, train_loader, val_loader, cfg)
    # outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    # save_to_file(outputs, out_fname) # Save the generated outputs to a file
