import main
import torch


def get_beer_categories(dataset):
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
    for row in tensor.split(1):
        num = torch.argmax(row).item() + 32  # correct for our shifted index.
        chars.append(chr(num))
    return ''.join(chars)


def beer2oh(beer):
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
    oh = torch.zeros((1, 104), dtype=torch.float)
    oh[0][index] = 1.0
    return oh


def oh2beer(tensor):
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


def scale_beer_rating(rating):
    """
    Scale ratings to be between -1 and 1
    :param rating: A float between 0. and 5.
    :return: tensor of the new rating
    """
    new_rating = ((rating * 2) / 5) - 1
    return torch.tensor([new_rating])  #TODO check if needs to be 2d for concatenation.


def get_metadata(row):
    beer = torch.squeeze(beer2oh(row['beer/style']))
    rating = scale_beer_rating(row['review/overall'])

    return torch.cat((beer, rating))


if __name__ == "__main__":
    data_dir = "../BeerAdvocatePA4"
    train_data_fname = data_dir + "/Beeradvocate_Train.csv"
    test_data_fname = data_dir + "/Beeradvocate_Test.csv"
    out_fname = ""

    train_data = main.load_data(train_data_fname) # Generating the pandas DataFrame

    print(get_beer_categories(train_data))
    print(get_metadata(train_data.iloc[2]))
