################################################################################
# Code author: Roger Milroy with credit to Jenny Hamer for code from which this is based.
# xray_dataloader.py from PA3
#
#
# Description:
# This code defines a custom PyTorch Dataset object suited for the
# BeerAdvocate dataset.
#
#
################################################################################

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Other libraries for data manipulation and visualization
import os
import numpy as np
import pandas as pd
import utilities


class BeerTrainDataset(Dataset):
    """
    Custom Dataset class for the BeerAdvocate Dataset.
    """

    def __init__(self, filename):

        self.data = pd.read_csv(filename)

    def __len__(self):

        # Return the total number of data samples
        return self.data['beer/style'].size

    def __getitem__(self, ind):
        """
        Gets a single item from the dataset.
        :param ind: the index of the item.
        :return: tuple of Tensors. Review text in one-hot encoding and metadata vector.
        """

        row = self.data.iloc[ind]
        text = utilities.char2oh(row['review/text'])
        metadata = utilities.get_metadata(row)

        # Return review and metadata
        return text, metadata


def create_split_loaders(batch_size, seed, filename, p_val=0.1, shuffle=True, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets.

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - filename: (string) path to the dataset.
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - extras: (dict)
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    """

    # Get create a ChestXrayDataset object
    dataset = BeerTrainDataset(filename)

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split:], all_indices[: val_split]

    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,
                            pin_memory=pin_memory)

    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader)