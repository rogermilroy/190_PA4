import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable

class baselineLSTM(nn.Module)
'''
Implementation of Hochreiter and Schmidhuber's LSTM
'''
    def __init__(self, config):
        super(baselineLSTM, self).__init__()

        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        self.input_dim = config.cfg['input_dim']
        self.hidden_dim = config.cfg['hidden_dim']
        self.layers = config.cfg['layers']
        self.dropout = config.cfg['dropout']
        self.bidirectional = config.cfg['bidirectional']
        self.batch_size = config.cfg['batch_size']

        # initialize lstm layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_state=self.hidden_dim,
                            num_layers=self.layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        # initialize output layer
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden(self):
        return Variable(torch.zeros(self.layers, self.batch_size, self.hidden_dim))

    def forward(self, sequence):
        # Takes in the sequence of the form (batch_size x sequence_length x input_dim) and
        # returns the output of form (batch_size x sequence_length x output_dim)

        hidden = init_hidden()
        out, hidden = self.lstm(sequence, hidden)
        return out

