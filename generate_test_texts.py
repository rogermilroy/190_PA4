from main import generate, process_test_data, save_to_file
from configs import *
from beer_dataloader import create_generation_loader
import sys
from models import *
from utilities import *
import torch

if __name__ == '__main__':

    data_dir = "/datasets/cs190f-public"
    test_data_fname = data_dir + "/Beeradvocate_Test.csv"

    modeltype = gen_cfg['model'].lower()
    if (modeltype == 'lstm' or modeltype == 'baselinelstm'):
        print("Using LSTM Model")
        model = baselineLSTM(gen_cfg)
    elif (modeltype == 'gru'):
        print("Using GRU Model")
        model = gru(gen_cfg)
    else:
        print("Model ", modeltype, " not found")
        sys.exit()

    # Check if your system supports CUDA
    use_cuda = torch.cuda.is_available()

    # Setup GPU optimization if CUDA is supported
    if use_cuda and gen_cfg['cuda']:
        computing_device = torch.device("cuda")
        extras = {"num_workers": 4, "pin_memory": True}
        print("CUDA is supported")
    else:  # Otherwise, train on the CPU
        computing_device = torch.device("cpu")
        extras = False
        print("CUDA NOT supported")

    model.load_state_dict(torch.load('./outputs/current_params_' + gen_cfg['model'] + '.pt',
                                                                         map_location='cpu'))
    model.to(computing_device)

    test_loader = create_generation_loader(gen_cfg['batch_size'], test_data_fname, extras=extras)

    texts = []
    with torch.no_grad():
        for minibatch_count, (beers, ratings) in enumerate(test_loader, 0):
            model.reset_hidden()
            batch = process_test_data(beers, ratings, computing_device)
            text = generate(model, batch, gen_cfg, computing_device)
            print(text[0])
            texts.extend(text)

    # val_samples = 0
    # bleu_score_avg = 0.
    # with torch.no_grad():
    #     for val_minibatch_count, (val_text, val_beer, val_rating) in enumerate(test_loader, 0):
    #         val_samples += len(val_beer)
    #
    #         # validation
    #         validation_loss = 0
    #         model.reset_hidden()
    #
    #
    #         # generate reviews and check bleu scores.
    #         generated_val_reviews = generate(model, process_test_data(val_beer,
    #                                                                   val_rating,
    #                                                                   computing_device),
    #                                          gen_cfg,
    #                                          computing_device)
    #         b_scores = torch.tensor(get_bleu_scores(generated_val_reviews, val_text))
    #         bleu_score_avg += float(torch.mean(b_scores))
    #         if val_samples > 2000:
    #             break
    #
    #     bleu_score_avg = (bleu_score_avg / float(val_minibatch_count))
    #     print("BLEU score: ", bleu_score_avg)

    # save_to_file(texts, './outputs/reviews_tau_' + gen_cfg['gen_temp'] + '_' + modeltype + '.txt')


