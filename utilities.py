import main
import torch
from nltk.translate import bleu_score
from torch.distributions import one_hot_categorical
from torch.nn.functional import softmax
import re
from beer_dataloader import *


def get_beer_categories(dataset):
    """
    Worker to find the set of beer styles in the dataset.
    :param dataset:
    :return:
    """
    styles = set()
    # loop through data
    for item in dataset['beer/style'].iteritems():
        # if we have seen it before add to set to keep track of new styles.
        if item[1] in styles:
            continue
        else:
            styles.add(item[1])
    beers = list(styles)
    beer_dict = {beers[i]: i for i in range(len(beers))}
    return beer_dict


def add_limiters(data):
    """
    Adds limiters and strips tab characters from review/text. To be run on a Pandas Dataframe
    before other processing.
    :param data: list of strings
    :return: list of strings
    """
    new = []
    for string in data:
        temp = string.replace('\t', ' ')
        temp = '^' + temp.strip() + '`'
        new.append(temp)
    return new


def find_longest(data):
    """
    Finds the length of the longest string in a list.
    :param data: List of strings.
    :return: int length of the longest string.
    """
    longest = 0
    for item in data:
        if len(item) > longest:
            longest = len(item)
    return longest


def pad_data(orig_data):
    """
    Pads a list or tuple of strings to be the same length after adding delimiters.
    :param orig_data: (list or tuple) the unpadded strings.
    :return: (list of strings) the padded strings.
    """
    delim = add_limiters(orig_data)
    # find the longest sequence
    max_length = find_longest(delim)
    # iterate and pad.
    padded = []
    for text in delim:
        difference = max_length - len(text)
        if difference > 0:
            temp = text + (difference * '`')
            padded.append(temp)
        else:
            padded.append(text)
    return padded


def strip_padding(texts):
    """
    Strips the SOS and EOS characters.
    :param texts: List of strings.
    :return: List of strings.
    """
    stripped = []
    for text in texts:
        stripped.append(re.sub('[\^`]', '', text))
    return stripped


def texts2oh(texts, computing_device):
    """
    Wrapper that takes a tuple or list of text, pads and converts to one-hot encoded form.
    :param texts: Tuple or list of strings. The texts to pad.
    :return: list of tensors. One hot encoded and padded.
    """
    padded = pad_data(texts)
    ohtexts = []
    for text in padded:
        oh = string2oh(text, computing_device)
        ohtexts.append(oh)
    return ohtexts


def string2oh(text, computing_device):
    """
    Converts a string to a one-hot encoded 2D Tensor.
    :param text: String of the review to be encoded.
    :return: list of Tensors. One hot encoded.
    """
    values = []
    # get ascii representation, create tensor and add one to index for each character
    for char in text:
        values.append(char2oh(char, computing_device))
    return torch.stack(values)


def char2oh(char, computing_device):
    index = ord(char) - 32  # we don't use 0-31 as non printable.
    temp = torch.zeros(98, dtype=torch.float, device=computing_device)
    temp[index] = 1.0
    return temp


def oh2texts(ohtexts):
    """
    Converts a list of lists of one-hot encoded letters into readable form.
    :param ohtexts: Tensor.
    :return: List of strings.
    """
    texts = []
    for text in ohtexts:
        texts.append(oh2string(text))
    return strip_padding(texts)


def oh2string(tensor):
    """
    Converts one-hot encoded values to a string.
    :param tensor: 2D Tensor of one-hot encoded values
    :return: String.
    """
    chars = []
    # iterate through tensor, get characters and add to list.
    for row in tensor:
        chars.append(oh2char(row))
    return ''.join(chars)


def oh2char(tensor):
    return chr(torch.argmax(tensor).item() + 32)


def beer2oh(beer, computing_device):
    """
    Converts string to one-hot encoding.
    :param beer: String. The beer to be encoded.
    :return: Tensor. One-hot encoded.
    """
    beers = {'American Double / Imperial Stout': 0, 'Euro Pale Lager': 1,
             'American Pale Wheat Ale': 2, 'Belgian Pale Ale': 3, 'Rye Beer': 4,
             'English Bitter': 5, 'Milk / Sweet Stout': 6, 'English Stout': 7, 'Kristalweizen': 8,
             'Roggenbier': 9, 'Dortmunder / Export Lager': 10, 'English Dark Mild Ale': 11,
             'Tripel': 12, 'Maibock / Helles Bock': 13, 'Smoked Beer': 14, 'American Black Ale': 15,
             'American Barleywine': 16, 'American Adjunct Lager': 17, 'Low Alcohol Beer': 18,
             'Quadrupel (Quad)': 19, 'American Porter': 20, 'Weizenbock': 21, 'Doppelbock': 22,
             'Berliner Weissbier': 23, 'Old Ale': 24, 'English Pale Ale': 25,
             'English Barleywine': 26, 'Czech Pilsener': 27, 'Flanders Oud Bruin': 28,
             'American Wild Ale': 29, 'California Common / Steam Beer': 30,
             'American Pale Lager': 31, 'Russian Imperial Stout': 32, 'Herbed / Spiced Beer': 33,
             'American Double / Imperial IPA': 34, 'English Strong Ale': 35,
             'Belgian Strong Dark Ale': 36, 'Vienna Lager': 37,
             'American Double / Imperial Pilsner': 38, 'Munich Dunkel Lager': 39,
             'American IPA': 40, 'Dunkelweizen': 41, 'Belgian Strong Pale Ale': 42, 'Faro': 43,
             'Gueuze': 44, 'American Strong Ale': 45, 'American Blonde Ale': 46, 'Dubbel': 47,
             'Extra Special / Strong Bitter (ESB)': 48, 'Lambic - Fruit': 49, 'Wheatwine': 50,
             'Keller Bier / Zwickel Bier': 51, 'Irish Red Ale': 52, 'Munich Helles Lager': 53,
             'Altbier': 54, 'Pumpkin Ale': 55, 'German Pilsener': 56, 'Chile Beer': 57,
             'Baltic Porter': 58, 'Eisbock': 59, 'Braggot': 60, 'Kölsch': 61,
             'Euro Strong Lager': 62, 'Lambic - Unblended': 63, 'Schwarzbier': 64,
             'Belgian IPA': 65, 'Witbier': 66, 'Hefeweizen': 67, 'Bock': 68,
             'Fruit / Vegetable Beer': 69, 'American Stout': 70, 'American Malt Liquor': 71,
             'Scotch Ale / Wee Heavy': 72, 'Japanese Rice Lager': 73,
             'Bière de Champagne / Bière Brut': 74, 'Winter Warmer': 75, 'Rauchbier': 76,
             'Light Lager': 77, 'American Amber / Red Ale': 78, 'English Brown Ale': 79,
             'Bière de Garde': 80, 'Gose': 81, 'Märzen / Oktoberfest': 82, 'Kvass': 83,
             'American Amber / Red Lager': 84, 'Irish Dry Stout': 85, 'Cream Ale': 86,
             'English India Pale Ale (IPA)': 87, 'Euro Dark Lager': 88, 'Scottish Ale': 89,
             'English Porter': 90, 'Flanders Red Ale': 91, 'Oatmeal Stout': 92,
             'American Dark Wheat Ale': 93, 'English Pale Mild Ale': 94, 'Belgian Dark Ale': 95,
             'American Brown Ale': 96, 'Black & Tan': 97, 'Sahti': 98, 'Foreign / Export Stout': 99,
             'Happoshu': 100, 'American Pale Ale (APA)': 101, 'Saison / Farmhouse Ale': 102,
             'Scottish Gruit / Ancient Herbed Ale': 103}
    index = beers[beer]
    oh = torch.zeros(104, dtype=torch.float, device=computing_device)
    oh[index] = 1.0
    return oh


def oh2beer(tensor):
    """
    Converts one-hot encoded vector to string.
    :param tensor: Tensor. One-hot encoded representation of the beer.
    :return: String. The beer.
    """
    beers = {'American Double / Imperial Stout': 0, 'Euro Pale Lager': 1,
             'American Pale Wheat Ale': 2, 'Belgian Pale Ale': 3, 'Rye Beer': 4,
             'English Bitter': 5, 'Milk / Sweet Stout': 6, 'English Stout': 7, 'Kristalweizen': 8,
             'Roggenbier': 9, 'Dortmunder / Export Lager': 10, 'English Dark Mild Ale': 11,
             'Tripel': 12, 'Maibock / Helles Bock': 13, 'Smoked Beer': 14, 'American Black Ale': 15,
             'American Barleywine': 16, 'American Adjunct Lager': 17, 'Low Alcohol Beer': 18,
             'Quadrupel (Quad)': 19, 'American Porter': 20, 'Weizenbock': 21, 'Doppelbock': 22,
             'Berliner Weissbier': 23, 'Old Ale': 24, 'English Pale Ale': 25,
             'English Barleywine': 26, 'Czech Pilsener': 27, 'Flanders Oud Bruin': 28,
             'American Wild Ale': 29, 'California Common / Steam Beer': 30,
             'American Pale Lager': 31, 'Russian Imperial Stout': 32, 'Herbed / Spiced Beer': 33,
             'American Double / Imperial IPA': 34, 'English Strong Ale': 35,
             'Belgian Strong Dark Ale': 36, 'Vienna Lager': 37,
             'American Double / Imperial Pilsner': 38, 'Munich Dunkel Lager': 39,
             'American IPA': 40, 'Dunkelweizen': 41, 'Belgian Strong Pale Ale': 42, 'Faro': 43,
             'Gueuze': 44, 'American Strong Ale': 45, 'American Blonde Ale': 46, 'Dubbel': 47,
             'Extra Special / Strong Bitter (ESB)': 48, 'Lambic - Fruit': 49, 'Wheatwine': 50,
             'Keller Bier / Zwickel Bier': 51, 'Irish Red Ale': 52, 'Munich Helles Lager': 53,
             'Altbier': 54, 'Pumpkin Ale': 55, 'German Pilsener': 56, 'Chile Beer': 57,
             'Baltic Porter': 58, 'Eisbock': 59, 'Braggot': 60, 'Kölsch': 61,
             'Euro Strong Lager': 62, 'Lambic - Unblended': 63, 'Schwarzbier': 64,
             'Belgian IPA': 65, 'Witbier': 66, 'Hefeweizen': 67, 'Bock': 68,
             'Fruit / Vegetable Beer': 69, 'American Stout': 70, 'American Malt Liquor': 71,
             'Scotch Ale / Wee Heavy': 72, 'Japanese Rice Lager': 73,
             'Bière de Champagne / Bière Brut': 74, 'Winter Warmer': 75, 'Rauchbier': 76,
             'Light Lager': 77, 'American Amber / Red Ale': 78, 'English Brown Ale': 79,
             'Bière de Garde': 80, 'Gose': 81, 'Märzen / Oktoberfest': 82, 'Kvass': 83,
             'American Amber / Red Lager': 84, 'Irish Dry Stout': 85, 'Cream Ale': 86,
             'English India Pale Ale (IPA)': 87, 'Euro Dark Lager': 88, 'Scottish Ale': 89,
             'English Porter': 90, 'Flanders Red Ale': 91, 'Oatmeal Stout': 92,
             'American Dark Wheat Ale': 93, 'English Pale Mild Ale': 94, 'Belgian Dark Ale': 95,
             'American Brown Ale': 96, 'Black & Tan': 97, 'Sahti': 98, 'Foreign / Export Stout': 99,
             'Happoshu': 100, 'American Pale Ale (APA)': 101, 'Saison / Farmhouse Ale': 102,
             'Scottish Gruit / Ancient Herbed Ale': 103}
    index = torch.argmax(tensor)
    beer_style = None
    for beer, i in beers.items():
        if i == index:
            beer_style = beer
            break
    return beer_style


def scale_beer_rating(rating, computing_device):
    """
    Scale ratings to be between -1 and 1
    :param rating: A float between 0. and 5.
    :return: tensor of the new rating
    """
    new_rating = ((rating * 2) / 5) - 1
    return torch.tensor([new_rating], dtype=torch.float, device=computing_device)


def get_metadata(beer, rating, computing_device):
    """
    Wrapper, converts string and float to metadata feature vector.
    :param beer:
    :param rating:
    :param computing_device
    :return:
    """
    beer = beer2oh(beer, computing_device)
    rating = scale_beer_rating(rating, computing_device)
    return torch.cat((beer, rating))


def get_metadatas(beers, ratings, computing_device):
    """
    Wrapper method for get_metadata that handles lists.
    :param beers: List of beers.
    :param ratings: List of ratings.
    :param computing_device
    :return: List of tensors. In encoded form.
    """
    metadatas = []
    for i in range(len(beers)):
        metadata = get_metadata(beers[i], ratings[i], computing_device)
        metadatas.append(metadata)
    return metadatas


def cat_batch_data(letter, metadata):
    if letter.size()[1] != metadata.size()[1]:
        cat = torch.cat((letter.permute(1, 0), metadata.permute(1, 0)))
        return cat.permute(1, 0)
    else:
        cat = torch.cat((letter, metadata))
        return cat


def concat_sequence_metadata(text, metadata):
    """
    Concatenates every character with the metadata.
    :param text: List of tensors. The review text.
    :param metadata: Tensor. The metadata.
    :return: List of tensors. The concatenated data.
    """
    seq_len = text.size()[0]
    meta = metadata.repeat(seq_len, 1)
    if text.size()[1] != meta.size()[1]:
        cat = torch.cat((text.permute(1, 0), meta.permute(1, 0)))
        return cat.permute(1, 0)
    else:
        cat = torch.cat((text, metadata))
        return cat


def concat_metadatas(texts, metadatas):
    """
    Wrapper for concat_metadata to handle lists
    :param texts:
    :param metadatas:
    :return:
    """
    concatenated = []
    for i in range(len(texts)):
        concatenated.append(concat_sequence_metadata(texts[i], metadatas[i]))
    return torch.stack(concatenated)


def batch2sequence(tensor):
    return tensor.permute(1, 0, 2)


def sequence2batch(sequence_list):
    return sequence_list.permute(1, 0, 2)


def to_indices(targets):
    t = targets.permute(1,0)[:98]
    return torch.argmax(t, 0)


def get_bleu_scores(outputs, targets):
    """
    Computes the bleu score for a list of network outputs and the corresponding target review.
    :param outputs: A list of strings, the network outputs.
    :param targets: A list of strings, the target review.
    :return:
    """
    scores = []
    for i in range(len(outputs)):
        score = bleu_score.corpus_bleu([[targets[i].split()]], [outputs[i].split()])
        scores.append(float(score))
    return scores


def all_finished(letters, computing_device):
    for letter in letters:
        if to_index(letter) != to_index(char2oh('`', computing_device)):
            # if we find a letter that isn't the EOS char we are not done.
            return False
    return True


def get_predicted_letters(outputs):
    """
    Sample from the distributions from the network.
    TODO need to add temperature to softmax.
    :param distributions: 2d tensor. Output from network.
    :return: List of tensors. The predicted letters in one hot encoding.
    """
    distributions = softmax(outputs, 1)

    sampler = one_hot_categorical.OneHotCategorical(distributions)
    prediction = sampler.sample()

    return prediction


if __name__ == "__main__":
    data_dir = "../BeerAdvocatePA4"
    train_data_fname = data_dir + "/Beeradvocate_Train.csv"
    test_data_fname = data_dir + "/Beeradvocate_Test.csv"
    out_fname = ""

    # train_loader, val_loader = create_split_loaders(2, 42, train_data_fname)
    # text1, beers1, rating1 = iter(train_loader).next()
    # print(get_bleu_scores(['Hello there, this is a test a big one'], ['Hello there, this is a '
    #                                                                    'test '
    #                                                                'run not a big one.']))
    #
    # data = to_tensor(texts2oh(text1))
    # print(data)
    # print(data.size())
    #
    # reshaped = batch2sequence(data)
    # print(reshaped)
    # print(reshaped.size())
    # print(strip_padding(['^Hello thersr, `````', '^^there wasa amistake`']))
    #
    # test_out = torch.tensor([[25., 32., -10., 17.]])
    # print(get_predicted_letters(test_out))

    a = torch.tensor([[1., 2., 3.], [3., 2., 1.]])
    b = torch.tensor([[4., 4.], [5., 5.]])

    c = concat_sequence_metadata(a, b)
    print(c)

def save_as_csv(training_losses, validation_losses, bleu_scores, cfg):
    # Save directories to use
    training_losses_dir = cfg['training_losses_dir'] + "_" +cfg['model'] + ".csv"
    validation_losses_dir = cfg['validation_losses_dir'] + "_" +cfg['model'] + ".csv"
    bleu_scores_dir = cfg['bleu_scores_dir'] + "_" +cfg['model'] + ".csv"

    # Convert the arrays as pandas dataframes
    training_losses_df = pd.DataFrame(training_losses)
    validation_losses_df = pd.DataFrame(validation_losses)
    bleu_scores_df = pd.DataFrame(bleu_scores)

    # Export dataframes to csv
    training_losses_df.to_csv(training_losses_dir)
    validation_losses_df.to_csv(validation_losses_dir)
    bleu_scores_df.to_csv(bleu_scores_dir)
