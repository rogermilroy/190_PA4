import main


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


if __name__ == "__main__":
    data_dir = "../BeerAdvocatePA4"
    train_data_fname = data_dir + "/Beeradvocate_Train.csv"
    test_data_fname = data_dir + "/Beeradvocate_Test.csv"
    out_fname = ""

    train_data = main.load_data(train_data_fname) # Generating the pandas DataFrame

    print(get_beer_categories(train_data))
