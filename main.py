import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from models import *
from configs import cfg
from utilities import *
from beer_dataloader import *
import time
import pandas as pd


def process_train_data(texts, beers, ratings, computing_device, character_only=False):
    """
    Processes a minibatch into one-hot encoding ready for input to the network.
    :param texts: A minibatch of reviews as a tuple.
    :param beers: A minibatch of beers as a tuple
    :param ratings: A minibatch of ratings as a tuple.
    :param computing_device: The device the tensors should be on.
    :param character_only: True if we are only training the language model.
    :return: tensor dims (s x N x d) N is minibatch size, s is sequence length and d is 98 if
    char_only or 203 otherwise.
    """
    data = texts2oh(texts, computing_device)
    # if we are not just training the language model.
    if not character_only:
        # concatenate text and metadata.
        metadatas = get_metadatas(beers, ratings, computing_device)
        data = concat_metadatas(data, metadatas)
    return batch2sequence(data)


def process_test_data(beers, ratings, computing_device):
    """
    Processesd a minibatch of test data into one hot encoding.
    :param beers: A minibatch of beers as a tuple.
    :param ratings: A minibatch of ratings as a tuple.
    :return: Tensor dims (N x d)
    """
    data = get_metadatas(beers, ratings, computing_device)
    return torch.stack(data)


def train(model, train_loader, val_loader, cfg, computing_device):

    model.to(computing_device)

    num_epochs = cfg['epochs']
    save_every = 445
    learning_rate = cfg['learning_rate']
    batch_size = cfg['batch_size']

    # use adam optimizer with default params and given learning rate
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    # use cross entropy loss as loss function
    criterion = nn.CrossEntropyLoss()

    # save important stats
    start_time = time.time()
    try:
        trng = pd.read_csv(open(cfg['training_losses_dir'] + '_' + cfg['model'] + '.csv'))
        vl = pd.read_csv(open(cfg['validation_losses_dir'] + '_' + cfg['model'] + '.csv'))
        bl = pd.read_csv(open(cfg['bleu_scores_dir'] + '_' + cfg['model'] + '.csv'))
        trng = trng.drop(columns=['Unnamed: 0'])
        vl = vl.drop(columns=['Unnamed: 0'])
        bl = bl.drop(columns=['Unnamed: 0'])
        training_losses = trng['0'].tolist()
        validation_losses = vl['0'].tolist()
        bleu_scores = bl['0'].tolist()

        df = pd.DataFrame(training_losses)
    except Exception as e:
        training_losses = []
        validation_losses = []
        bleu_scores = []
        print(e)

    training_loss_avg = 0
    best_val = 1000000.
    best_params = None

    for epoch in range(num_epochs):
        print("Epoch: " + str(epoch))

        # Get next minibatch of data for training
        torch.cuda.empty_cache()
        for minibatch_count, (text, beer, rating) in enumerate(train_loader, 0):
            print("On minibatch: ", minibatch_count)

            batch = process_train_data(text, beer, rating, computing_device)

            # training
            model.zero_grad()
            model.reset_hidden()
            training_loss = 0.
            for c in range(batch.size()[0]):
                tens = torch.unsqueeze(batch[c], 0)
                output = model(tens)
                if c < batch.size()[0] - 1:
                    targets = to_indices(batch[c+1])
                else:
                    targets = to_indices(batch[c])
                training_loss += criterion(output, targets)

            training_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # calculate loss
            minibatch_loss_avg = (training_loss / float(batch.size()[0])).item()
            training_loss_avg += minibatch_loss_avg
            print("Minibatch Loss Avg: ", minibatch_loss_avg)

            # Calculate validation of every plot_every minibatches
            if minibatch_count % save_every == 0 and minibatch_count != 0:
                print("Validation: ", (minibatch_count / save_every))

                # add the average training loss to an array to plot later.
                training_loss_avg = training_loss_avg / float(save_every)
                training_losses.append(training_loss_avg)
                print("Training Loss Avg: ", training_loss_avg)
                training_loss_avg = 0.
                bleu_score_avg = 0.
                validation_loss_avg = 0.

                val_samples = 0.
                # Get next minibatch of data for validation
                torch.cuda.empty_cache()
                with torch.no_grad():
                    for val_minibatch_count, (val_text, val_beer, val_rating) in enumerate(val_loader, 0):
                        val_batch = process_train_data(val_text, val_beer, val_rating, computing_device)
                        val_batch.to(computing_device)
                        val_samples += batch_size

                        # validation
                        validation_loss = 0
                        model.reset_hidden()
                        for c in range(val_batch.size()[0]):
                            val_tens = torch.unsqueeze(val_batch[c], 0)
                            val_output = model(val_tens)
                            if c < len(val_text) - 1:
                                val_targets = to_indices(val_batch[c + 1])
                            else:
                                val_targets = to_indices(val_batch[c])
                            validation_loss += float(criterion(val_output, val_targets))

                        # calculate loss per review
                        batch_loss_avg = (validation_loss / float(batch.size()[0]))
                        validation_loss_avg += batch_loss_avg

                        # generate reviews and check bleu scores.
                        generated_val_reviews = generate(model, process_test_data(val_beer,
                                                                                  val_rating,
                                                                                  computing_device),
                                                         cfg,
                                                         computing_device)
                        b_scores = torch.tensor(get_bleu_scores(generated_val_reviews, val_text))
                        bleu_score_avg += float(torch.mean(b_scores))

                    # add average loss over validation set to array
                    validation_loss_avg = validation_loss_avg / float(val_samples)
                    validation_losses.append(validation_loss_avg)
                    bleu_score_avg = (bleu_score_avg / float(val_samples))
                    bleu_scores.append(bleu_score_avg)
                    print("Validation Loss: ", validation_loss_avg)
                    print("BLEU score: ", bleu_score_avg)

                    # keep best parameters so far and keep track of them.
                    if validation_loss_avg < best_val:
                        print("Updated params!")
                        best_val = validation_loss_avg
                        best_params = model.state_dict()
                        torch.save(best_params, './outputs/best_params.pt')
                    torch.save(model.state_dict(), './outputs/current_params.pt')
                    save_as_csv(training_losses, validation_losses, bleu_scores, cfg)

    return training_losses, validation_losses, bleu_scores, best_params


def generate(model, batch, cfg, computing_device):
    """
    Given n rows in test data, generate a list of n strings, where each string is the review
    corresponding to each input row in test data.
    :param model:
    :param batch:
    :param cfg:
    :return:
    """
    # Initialise a list of SOS characters.
    letters = torch.stack([char2oh('^', computing_device) for i in range(len(batch))])
    gen_texts = []

    # Loop until only EOS is predicted.
    while not all_finished(letters, computing_device) and len(gen_texts) < cfg['max_len']:
        inp = cat_batch_data(letters, batch)
        outputs = model.forward(torch.unsqueeze(inp, 0))
        # sample from softmax distribution.
        letters = get_predicted_letters(outputs, cfg['gen_temp'])
        gen_texts.append(letters)
    # convert to strings and return.
    return oh2texts(sequence2batch(torch.stack(gen_texts)))


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


    modeltype = cfg['model'].lower()
    if(modeltype == 'lstm' or modeltype == 'baselinelstm'):
        print("Using LSTM Model")
        model = baselineLSTM(cfg)
    elif(modeltype == 'gru'):
        print("Using GRU Model")
        model = gru(cfg)
    else:
        print("Model ", modeltype, " not found")
        sys.exit()

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda and cfg['cuda']:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 4, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    model.load_state_dict(torch.load('./outputs/current_params.pt'))

    train_loader, val_loader = create_split_loaders(cfg['batch_size'], 42, train_data_fname,
                                                    extras=extras, subset=True, p_val=0.1)

    training_losses, validation_losses, bleu_scores, params = train(model, train_loader,
                                                                    val_loader,
                                                             cfg, computing_device)
    torch.save(params, cfg['params_dir'])
    save_as_csv(training_losses, validation_losses, bleu_scores, cfg)

    # outputs = generate(model, X_test, cfg) # Generate the outputs for test data
    # save_to_file(outputs, out_fname) # Save the generated outputs to a file
