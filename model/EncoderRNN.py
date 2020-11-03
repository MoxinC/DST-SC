import torch
import torch.nn as nn
from torch.nn import utils as nn_utils

class EncoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, rnn_layers):
        super(EncoderRNN, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.rnn_layers, batch_first=True, bidirectional=True)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return weight.new(self.rnn_layers * 2, bsz, self.hidden_size).zero_()

    def forward(self, x, x_length, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(len(x))
        x_embedding = nn_utils.rnn.pack_padded_sequence(x, x_length, batch_first=True)
        output, hidden = self.gru(x_embedding, hidden)
        output, _ = nn_utils.rnn.pad_packed_sequence(output, batch_first=True)
        hidden = hidden.view(self.rnn_layers, 2, -1, self.hidden_size)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        hidden = hidden[:, 0, :, :] + hidden[:, 1, :, :]
        return output, hidden
