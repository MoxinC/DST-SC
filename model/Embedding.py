import numpy as np
import torch
import torch.nn as nn


class Embedding(nn.Module):

    def __init__(self, args):
        super(Embedding, self).__init__()
        self.args = args
        self.word2index = {}
        self.index2word = {}
        self.load_vocab()

        self.encoder = nn.Embedding(len(self.word2index), self.args.embedding_size).to(self.args.device)

        self.init_weights()
        print("Initialized Embedding Layer")

    def load_vocab(self):
        words = ['PAD', 'UNK', 'SEP', 'EOS']
        with open(self.args.vocab, 'r', encoding='utf-8') as f:
            for line in f:
                words.append(line[:-1])
        self.word2index = {w: i for i, w in enumerate(words)}
        self.index2word = {i: w for i, w in enumerate(words)}

    def init_weights(self):
        if self.args.pretrained is None:
            return

        self.char_encoder = KazumaCharEmbedding(self.args.char_pretrained)
        word_embedding_size = self.args.embedding_size - self.args.char_embedding_size
        unknown_word_embedding = np.random.uniform(-0.1, 0.1, word_embedding_size).tolist()        

        # load pretrained word embeddings
        pretrained_word_embeddings = {}
        print("Load Pretrained Embeddings")
        with open(self.args.pretrained, encoding='utf-8') as f:
            for line in f:
                word = line.split(' ')[0]
                vector = line.split(' ')[1:]
                vector = np.array([float(dim) for dim in vector])
                pretrained_word_embeddings[word] = vector

        for word, index in self.word2index.items():
            if word in pretrained_word_embeddings:
                word_embedding = pretrained_word_embeddings[word]
            else:
                word_embedding = np.random.uniform(-0.1, 0.1, word_embedding_size).tolist()
            char_embedding = self.char_encoder.emb(word)
            embedding = np.concatenate((word_embedding, char_embedding))
            embedding = torch.Tensor(embedding)
            self.encoder.weight.data[index] = torch.Tensor(embedding)

        if self.args.fix_embedding:
            self.encoder.weight.requires_grad = False

    def forward(self, x):
        embed_sens = self.encoder(x)
        return embed_sens


"""
MIT License

Copyright (c) 2017 Victor Zhong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def ngrams(sentence, n):
    """
    Returns:
        list: a list of lists of words corresponding to the ngrams in the sentence.
    """
    return [sentence[i:i+n] for i in range(len(sentence)-n+1)]


class KazumaCharEmbedding():
    """
    Reference: http://www.logos.t.u-tokyo.ac.jp/~hassy/publications/arxiv2016jmt/
    """

    d_emb = 100

    def __init__(self, url):
        self.ngram2vec = {}
        self.load_word2emb(url)

    def emb(self, w):
        chars = ['#BEGIN#'] + list(w) + ['#END#']
        embs = np.zeros(self.d_emb, dtype=np.float32)
        match = {}
        for i in [2, 3, 4]:
            grams = ngrams(chars, i)
            for g in grams:
                g = '{}gram-{}'.format(i, ''.join(g))
                e = self.ngram2vec.get(g, None)
                if e is not None:
                    match[g] = np.array(e, np.float32)
        if match:
            embs = sum(match.values()) / len(match)
        return embs.tolist()

    def load_word2emb(self, url):
        with open(url, 'r') as f:
            for line in f:
                elems = line.split()
                word = elems[0]
                vector = [float(n) for n in elems[1:]]
                self.ngram2vec[word] = vector
