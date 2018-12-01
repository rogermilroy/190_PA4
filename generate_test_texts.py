from main import generate, process_test_data, save_to_file
import configs
from beer_dataloader import create_test_loader
import sys
from models import *
import torch

if __name__ == '__main__':

    data_dir = "../BeerAdvocateDataset"
    train_data_fname = data_dir + "/Beeradvocate_Train.csv"
    test_data_fname = data_dir + "/Beeradvocate_Test.csv"

    modeltype = cfg['model'].lower()
    if (modeltype == 'lstm' or modeltype == 'baselinelstm'):
        print("Using LSTM Model")
        model = baselineLSTM(cfg)
    elif (modeltype == 'gru'):
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

    test_loader = create_test_loader(cfg['batch_size'], 42, test_data_fname, extras=extras)

    texts = []

    for minibatch_count, beers, ratings in enumerate(test_loader, 0):
        batch = process_test_data(beers, ratings, computing_device)
        text = generate(model, batch, cfg, computing_device)
        for t in text:
            texts.append(t)

    save_to_file(texts, './outputs/' + modeltype + '-texts.txt')


