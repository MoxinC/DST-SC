import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, embedding, embedding_size, hidden_size, rnn_layers, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.vocab_size = vocab_size
        self.gru = nn.GRU(self.embedding_size, self.hidden_size, self.rnn_layers, batch_first=True)

    def forward(self, x, hidden):
        decoder_output, decoder_hidden = self.gru(x, hidden)
        return decoder_output, decoder_hidden
