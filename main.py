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

    num_epochs = cfg['epochs']
    save_every = 1000
    learning_rate = cfg['learning_rate']
    batch_size = cfg['batch_size']

    # use adam optimizer with default params and given learning rate
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # use cross entropy loss as loss function
    criterion = nn.CrossEntropyLoss()

    # save important stats
    start_time = time.time()
    training_losses = []
    validation_losses = []
    bleu_scores = []
    training_loss_avg = 0
    validation_loss_avg = 0
    best_val = 10000000
    best_params = None

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        # Get next minibatch of data for training
        torch.cuda.empty_cache()
        for minibatch_count, (text, beer, rating) in enumerate(train_loader, 0):
            print("On minibatch: ", minibatch_count)

            batch = process_train_data(text, beer, rating)

            # training
            model.zero_grad()
            model.reset_hidden()
            training_loss = 0
            for c in range(len(text)):
                tens = torch.unsqueeze(batch[c], 0)
                output = model(tens)
                if c < len(text) - 1:
                    targets = to_indices(batch[c+1])
                else:
                    targets = to_indices(get_terminating_batch(batch[c]))
                crit_inputs = torch.squeeze(output)
                training_loss += criterion(crit_inputs, targets)

            training_loss.backward()
            optimizer.step()

            # calculate loss
            training_loss_avg += (training_loss / batch_size).item()
            print((training_loss / batch_size).item())

            training_loss = 0

            # Calculate validation of every plot_every minibatches
            if minibatch_count % save_every == 0 and minibatch_count != 0:
                print("Validation ", (minibatch_count / save_every))

                # add the average training loss to an array to plot later.
                training_loss_avg = training_loss_avg / float(save_every)
                training_losses.append(training_loss_avg)
                print("Training loss: ", training_loss_avg)
                training_loss_avg = 0
                bleu_score_avg = 0
                validation_loss_avg = 0

                val_samples = 0
                # Get next minibatch of data for validation
                torch.cuda.empty_cache()
                for val_minibatch_count, (val_text, val_beer, val_rating) in enumerate(val_loader,
                                                                                       0):

                    val_batch = process_train_data(val_text, val_beer, val_rating)
                    val_samples += batch_size
                    # validation
                    validation_loss = 0
                    for c in range(len(val_text)):
                        val_tens = torch.unsqueeze(val_batch[c], 0)
                        val_output = model(val_tens)
                        if c < len(val_text) - 1:
                            val_targets = to_indices(val_batch[c + 1])
                        else:
                            val_targets = to_indices(get_terminating_batch(val_batch[c]))
                        val_crit_inputs = torch.squeeze(val_output)
                        validation_loss += float(criterion(val_crit_inputs, val_targets))

                    # calculate loss per review
                    validation_loss_avg += (validation_loss / float(batch_size))

                    # generate reviews and check bleu scores.
                    generated_val_reviews = generate(model, process_test_data(val_beer,
                                                                              val_rating), cfg)
                    bleu_scores = torch.tensor(get_bleu_scores(generated_val_reviews, val_text))
                    bleu_score_avg += torch.mean(bleu_scores)

                # add average loss over validation set to array
                validation_loss_avg /= float(val_samples)
                validation_losses.append(validation_loss_avg)
                bleu_score_avg = (bleu_score_avg / float(val_samples)).item()
                bleu_scores.append(bleu_score_avg)
                print("Validation Loss: ", validation_loss_avg)
                print("BLEU score: ", bleu_score_avg)

                # keep best parameters so far and keep track of them.
                if validation_loss_avg < best_val:
                    print("Updated params!")
                    best_val = validation_loss_avg
                    best_params = model.state_dict()

                # break if loss goes up too many times consecutively
                if (False):
                    # TODO BREAK AFTER VALIDATION LOSS INCREASES
                    break;

    return training_losses, validation_losses, bleu_scores


def generate(model, batch, cfg):
    """
    Given n rows in test data, generate a list of n strings, where each string is the review
    corresponding to each input row in test data.
    :param model:
    :param batch:
    :param cfg:
    :return:
    """
    # Initialise a list of SOS characters.
    letters = [char2oh('^') for i in range(len(batch))]
    gen_texts = []
    list_batch = list(torch.split(batch, 1))

    # Loop until only EOS is predicted.
    while not all_finished(letters) and len(gen_texts) < cfg['max_len']:
        inp = cat_batch_data(letters, list_batch)
        outputs = torch.squeeze(model.forward(torch.unsqueeze(inp, 0)))
        # sample from softmax distribution.
        letters = get_predicted_letters(outputs)
        gen_texts.append(letters)
    # convert to strings and return.
    return oh2texts(sequence2batch(gen_texts))


def save_to_file(outputs, fname):
    # TODO: Given the list of generated review outputs and output file name, save all these reviews to
    # the file in .txt format.
    with open(fname, 'w') as file:
        for output in outputs:
            file.write(output + '\n')  # TODO test!


if __name__ == "__main__":
    data_dir = "../BeerAdvocateDataset"
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

    train_loader, val_loader = create_split_loaders(cfg['batch_size'], 42, train_data_fname,
                                                    subset=True)

    model = baselineLSTM(cfg) # Replace this with model = <your model name>(cfg)
    if cfg['cuda']:
        computing_device = torch.device("cuda")
    else:
        computing_device = torch.device("cpu")
    model.to(computing_device)

    train(model, train_loader, val_loader, cfg)

    # outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    # save_to_file(outputs, out_fname) # Save the generated outputs to a file
