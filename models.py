import torch
import torch.nn as nn
import torch.nn.init as torch_init
from torch.autograd import Variable
from configs import cfg

class baselineLSTM(nn.Module):
    '''
    Implementation of Hochreiter and Schmidhuber's LSTM
    '''
    def __init__(self, config):
        super(baselineLSTM, self).__init__()

        # Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.output_dim = cfg['output_dim']
        self.layers = cfg['layers']
        self.dropout = cfg['dropout']
        self.bidirectional = cfg['bidirectional']
        self.batch_size = cfg['batch_size']
        self.cell_state = None
        self.hidden_state = None

        # initialize lstm layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.recurrent_normed = nn.BatchNorm1d(self.hidden_dim)

        # initialize output layer
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.xavier_normal_(self.hidden2out.weight)

    def reset_hidden(self):
        self.cell_state = None
        self.hidden_state = None

    def forward(self, sequence):
        # Takes in the sequence of the form (1 x batch_size x input_dim) and
        # returns the output of form (1 x batch_size x output_dim)

        hidden = (self.hidden_state, self.cell_state)
        if self.cell_state is None or self.hidden_state is None:
            out, states = self.lstm(sequence)
        else:
            out, states = self.lstm(sequence, hidden)
        self.hidden_state = states[0]
        self.cell_state = states[1]
        out = self.recurrent_normed(torch.squeeze(out))
        out = self.hidden2out(out)
        return out


class gru(nn.Module):
    def __init__(self, config):
        super(gru, self).__init__()

        #Initialize your layers and variables that you want;
        # Keep in mind to include initialization for initial hidden states of LSTM, you
        # are going to need it, so design this class wisely.
        self.input_dim = cfg['input_dim']
        self.hidden_dim = cfg['hidden_dim']
        self.output_dim = cfg['output_dim']
        self.layers = cfg['layers']
        self.dropout = cfg['dropout']
        self.bidirectional = cfg['bidirectional']
        self.batch_size = cfg['batch_size']
        self.hidden_state = None

        # initialize GRU layer
        self.gru = nn.GRU(input_size=self.input_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=self.layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        self.recurrent_normed = nn.BatchNorm1d(self.hidden_dim)

        # initialize output layer
        self.hidden2out = nn.Linear(self.hidden_dim, self.output_dim)
        nn.init.xavier_normal_(self.hidden2out.weight)

    def reset_hidden(self):
        self.hidden_state = None

    def forward(self, sequence):
        # Takes in the sequence of the form (1 x batch_size x input_dim) and
        # returns the output of form (1 x batch_size x output_dim)

        if self.hidden_state is None:
            out, hidden = self.gru(sequence)
        else:
            out, hidden = self.gru(sequence, self.hidden_state)
        self.hidden_state = hidden

        # normalise
        out = self.recurrent_normed(torch.squeeze(out))

        out = self.hidden2out(out)
        return out


