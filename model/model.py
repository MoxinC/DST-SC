import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils

from model.Embedding import Embedding
from model.EncoderRNN import EncoderRNN
from model.DecoderRNN import DecoderRNN
from util.util import attend, masked_softmax, get_ontology


NUM_GATES = 3


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_gates = NUM_GATES
        self.embedding = Embedding(args)
        self.dropout = nn.Dropout(self.args.dropout)
        self.vocab = self.embedding.word2index
        self.index2word = self.embedding.index2word
        self.none_idx = self.vocab['none']
        self.dontcare_idx= self.vocab['dontcare']
        self.eos_idx= self.vocab['EOS']
        self.pad_idx= self.vocab['PAD']

        # load ontology
        ontology = get_ontology(self.args.ontology)
        self.domain_slots = list(ontology.keys())

        # init encoder and decoder
        self.input_encoder = EncoderRNN(args.embedding_size, args.hidden_size, args.rnn_layers)
        self.bs_decoder = DecoderRNN(self.embedding, args.embedding_size, args.hidden_size, args.rnn_layers, len(self.vocab))

        # domain-slot embedding
        self.domain_slot_w2i = {}
        for domain_slot in self.domain_slots:
            domain, slot = domain_slot.split('-')
            if domain not in self.domain_slot_w2i.keys():
                self.domain_slot_w2i[domain] = len(self.domain_slot_w2i)
            if slot not in self.domain_slot_w2i.keys():
                self.domain_slot_w2i[slot] = len(self.domain_slot_w2i)
        self.domain_slot_embedding = nn.Embedding(len(self.domain_slot_w2i), self.args.embedding_size).to(self.args.device)
        self.domain_slot_embedding.weight.data.uniform_(-0.1, 0.1)

        # slot gate
        self.slot_gate = nn.Linear(self.args.hidden_size, self.num_gates)
        self.slot_gate.weight.data.uniform_(-0.1, 0.1)

        # linears
        self.attn_w_linear = nn.Linear(self.args.hidden_size * 2, self.args.hidden_size)
        self.attn_w_linear.weight.data.uniform_(-0.1, 0.1)
        self.attn_e_linear = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.attn_e_linear.weight.data.uniform_(-0.1, 0.1)

        self.vocab_linear = nn.Linear(self.args.hidden_size * 2, len(self.vocab))
        self.vocab_linear.weight.data.uniform_(-0.1, 0.1)

        self.gen_linear = nn.Linear(self.args.hidden_size * 2 + self.args.embedding_size, 1)
        self.gen_linear.weight.data.uniform_(-0.1, 0.1)

        self.rel_attn_w_linear = nn.Linear(self.args.hidden_size + self.args.embedding_size, self.args.hidden_size)
        self.rel_attn_w_linear.weight.data.uniform_(-0.1, 0.1)
        self.rel_attn_e_linear = nn.Linear(self.args.hidden_size, 1, bias=False)
        self.rel_attn_e_linear.weight.data.uniform_(-0.1, 0.1)
        self.copy_rel_linear = nn.Linear(self.args.hidden_size, 1)
        self.copy_rel_linear.weight.data.uniform_(-0.1, 0.1)
        
        print("Initialized model")

    def get_domain_slot_embedding(self):
        all_domain_slot_emb = []
        for domain_slot in self.domain_slots:
            domain, slot = domain_slot.split('-')
            if domain in self.domain_slot_w2i.keys():
                domain_w2idx = [self.domain_slot_w2i[domain]]
                domain_w2idx = torch.LongTensor(domain_w2idx).to(self.args.device)
                domain_emb = self.domain_slot_embedding(domain_w2idx)
            if slot in self.domain_slot_w2i.keys():
                slot_w2idx = [self.domain_slot_w2i[slot]]
                slot_w2idx = torch.LongTensor(slot_w2idx).to(self.args.device)
                slot_emb = self.domain_slot_embedding(slot_w2idx)
            all_domain_slot_emb.append(domain_emb + slot_emb)
        all_domain_slot_emb = torch.stack(all_domain_slot_emb, 0).squeeze(1).to(self.args.device)
        return all_domain_slot_emb

    def pad_and_sort_x(self, x):
        batch_size = len(x)
        lengths = np.array([max(1, len(one)) for one in x])
        max_length = max(lengths)
        padded_x = np.full((batch_size, max_length), self.pad_idx)
        for i, one in enumerate(x):
            padded_x[i, :len(one)] = one

        sort_index = np.argsort(-lengths)
        unsorted_index = np.argsort(sort_index)
        sorted_padded_x = padded_x[sort_index]
        sorted_lengths = lengths[sort_index]

        sorted_padded_x = torch.Tensor(sorted_padded_x).type(torch.LongTensor).to(self.args.device)
        return sorted_padded_x, unsorted_index, sorted_lengths

    def pad_y(self, y, max_length=0):
        batch_size = len(y)
        num_slots = len(y[0])

        lengths = np.zeros((num_slots, batch_size), dtype=int)
        for i in range(num_slots):
            for j, one in enumerate(y):
                one_i_len = len(one[i])
                max_length = max(one_i_len, max_length)
                lengths[i, j] = one_i_len

        padded_y = np.full((num_slots, batch_size, max_length), self.pad_idx)
        for i in range(num_slots):
            for j, one in enumerate(y):
                padded_y[i, j, :len(one[i])] = one[i]

        padded_y = torch.Tensor(padded_y).type(torch.LongTensor).to(self.args.device)
        lengths = torch.Tensor(lengths).type(torch.LongTensor).to(self.args.device)
        return padded_y, lengths

    def forward(self, x, y, last_y):
        pad_x, unsorted_index, x_lengths = self.pad_and_sort_x(x)
        all_turns_num, max_x_length = pad_x.size()

        # word dropout
        if self.args.word_dropout > 0:
            rand_mask = np.random.binomial([np.ones((all_turns_num, max_x_length))], 1 - self.args.word_dropout)[0]
            rand_mask = torch.LongTensor(rand_mask).to(self.args.device)
            pad_x = pad_x * rand_mask

        pad_y, y_lengths = self.pad_y(y)
        max_decode_length = pad_y.size(2)
        pad_last_y, _ = self.pad_y(last_y, max_decode_length)

        # encoder
        x_embedding = self.embedding(pad_x)
        x_embedding = self.dropout(x_embedding)
        encoder_all_h, encoder_ht = self.input_encoder(x_embedding, x_lengths)
        encoder_all_h = encoder_all_h[unsorted_index]
        encoder_ht = encoder_ht[:, unsorted_index, :]
        pad_x = pad_x[unsorted_index]
        x_lengths = x_lengths[unsorted_index]

        # decoder initialize
        all_decoder_output = torch.zeros(len(self.domain_slots), all_turns_num, max_decode_length, len(self.vocab)).to(self.args.device)
        all_gate_output = torch.zeros(len(self.domain_slots), all_turns_num, self.num_gates).to(self.args.device)

        # get domain-slot embedding
        all_domain_slot_emb = self.get_domain_slot_embedding()
        
        for domain_slot_index in range(len(self.domain_slots)):
            # first decoder input: domain slot embedding 
            decoder_input = all_domain_slot_emb[domain_slot_index].view(1, 1, -1).repeat(all_turns_num, 1, 1)
            decoder_hidden = encoder_ht
            target = pad_y[domain_slot_index]
            
            # mask for last_y
            mask_not_none_last_y = (pad_last_y[:, :, 0] != self.none_idx).float()
            mask_rel_last_y = torch.ones(len(self.domain_slots), all_turns_num).to(self.args.device)
            mask_rel_last_y[domain_slot_index, :] = 0
            mask_rel_last_y = mask_not_none_last_y * mask_rel_last_y
            mask_rel_last_y = mask_rel_last_y.transpose(0, 1)

            for decode_idx in range(max_decode_length):
                decoder_output, decoder_hidden = self.bs_decoder(decoder_input, decoder_hidden)
                decoder_output = decoder_output.squeeze(1)

                # word copying distribution
                scores = self.attn_e_linear(torch.tanh(self.attn_w_linear(
                    torch.cat((encoder_all_h, decoder_output.unsqueeze(1).expand_as(encoder_all_h)), dim=-1))))
                scores = scores.squeeze(-1)
                max_len = max(x_lengths)
                for i, l in enumerate(x_lengths):
                    if l < max_len:
                        scores.data[i, l:] = -np.inf
                scores = F.softmax(scores, dim=-1)
                context = scores.unsqueeze(2).expand_as(encoder_all_h).mul(encoder_all_h).sum(1)

                # generative dist
                gen_dist = F.softmax(self.vocab_linear(torch.cat((decoder_output, context), dim=-1)), dim=1)
                
                # soft gate g_1
                gen_prob = torch.sigmoid(self.gen_linear(torch.cat((decoder_output, context, decoder_input.squeeze(1)), -1)))

                # value copying distribution
                if decode_idx == 0:
                    rel_scores = self.rel_attn_e_linear(torch.tanh(self.rel_attn_w_linear(
                        torch.cat((all_domain_slot_emb.unsqueeze(0).repeat(all_turns_num, 1, 1),
                                   decoder_output.unsqueeze(1).repeat(1, len(self.domain_slots), 1)), dim=-1)
                    )))
                    rel_scores = masked_softmax(rel_scores.squeeze(-1), mask_rel_last_y)
                    
                    # soft gate g_2
                    copy_rel_prob = torch.sigmoid(self.copy_rel_linear(context))

                    # slot gate
                    gate_output = self.slot_gate(context)
                    all_gate_output[domain_slot_index, :, :] = gate_output

                # final distribution
                final_dist = (1 - copy_rel_prob) * gen_prob * gen_dist
                final_dist.scatter_add_(1, pad_x, (1 - copy_rel_prob) * (1 - gen_prob) * scores)
                final_dist.scatter_add_(1, pad_last_y[:, :, decode_idx].transpose(0, 1), copy_rel_prob * rel_scores)
                all_decoder_output[domain_slot_index, :, decode_idx, :] = final_dist

                # get next input
                top_idx = torch.argmax(final_dist, dim=1)
                if self.args.use_teacher_forcing and random.random() < self.args.teacher_forcing_ratio:
                    next_input_idx = target[:, decode_idx]
                else:
                    next_input_idx = top_idx
                decoder_input = self.embedding(next_input_idx.unsqueeze(-1))
        
        return all_decoder_output, pad_y, y_lengths, all_gate_output

    def test(self, x, gate_y):
        flat_x = sum(x, [])
        pad_x, unsorted_index, x_lengths = self.pad_and_sort_x(flat_x)
        all_turns_num = pad_x.size()[0]
    
        # encoder
        x_embedding = self.embedding(pad_x)
        encoder_all_h, encoder_ht = self.input_encoder(x_embedding, x_lengths)
        encoder_all_h = encoder_all_h[unsorted_index]
        encoder_ht = encoder_ht[:, unsorted_index, :]
        pad_x = pad_x[unsorted_index]
        x_lengths = x_lengths[unsorted_index]
    
        # decoder
        max_decode_length = self.args.max_decode_length
        all_gate_output = torch.zeros(len(self.domain_slots), all_turns_num, self.num_gates).to(self.args.device)
        all_decoded_index = torch.zeros(len(self.domain_slots), all_turns_num, max_decode_length)
        all_decoded_index = all_decoded_index.type(torch.LongTensor).to(self.args.device)
    
        # get domain-slot embedding
        all_domain_slot_emb = self.get_domain_slot_embedding()
        
        # init last_y
        last_y = np.tile(np.array([self.none_idx, self.eos_idx]), (1, len(self.domain_slots), 1))
        last_y, _ = self.pad_y(last_y, max_length=max_decode_length)
    
        for turn_idx in range(all_turns_num):
            turn_encoder_all_h = encoder_all_h[turn_idx].unsqueeze(0)
            turn_encoder_ht = encoder_ht[:, turn_idx, :].unsqueeze(1)
            turn = pad_x[turn_idx].unsqueeze(0)
            turn_length = [x_lengths[turn_idx], ]
    
            new_y = []
            for domain_slot_index in range(len(self.domain_slots)):
                decoder_input = all_domain_slot_emb[domain_slot_index].view(1, 1, -1)
                decoder_hidden = turn_encoder_ht
                decoded_index = []
    
                # mask for last_y
                mask_not_none_last_y = (last_y[:, :, 0] != self.none_idx).float()
                mask_rel_last_y = torch.ones(len(self.domain_slots), all_turns_num).to(self.args.device)
                mask_rel_last_y[domain_slot_index, :] = 0
                mask_rel_last_y = mask_not_none_last_y * mask_rel_last_y
                mask_rel_last_y = mask_rel_last_y.transpose(0, 1)
    
                for decode_idx in range(max_decode_length):
                    decoder_output, decoder_hidden = self.bs_decoder(decoder_input, decoder_hidden)
                    decoder_output = decoder_output.squeeze(1)
    
                    # word copying distribution
                    scores = self.attn_e_linear(torch.tanh(self.attn_w_linear(
                        torch.cat((turn_encoder_all_h, decoder_output.unsqueeze(1).expand_as(turn_encoder_all_h)), dim=-1))))
                    scores = scores.squeeze(-1)
                    max_len = max(x_lengths)
                    for i, l in enumerate(turn_length):
                        if l < max_len:
                            scores.data[i, l:] = -np.inf
                    scores = F.softmax(scores, dim=-1)
                    context = scores.unsqueeze(2).expand_as(turn_encoder_all_h).mul(turn_encoder_all_h).sum(1)
    
                    # generative distribution
                    gen_dist = F.softmax(self.vocab_linear(torch.cat((decoder_output, context), dim=-1)), dim=1)
                    
                    # soft gate g_1
                    gen_prob = torch.sigmoid(self.gen_linear(torch.cat((decoder_output, context, decoder_input.squeeze(1)), -1)))

                    # value copying distribution
                    if decode_idx == 0:
                        rel_scores = self.rel_attn_e_linear(torch.tanh(self.rel_attn_w_linear(
                            torch.cat((all_domain_slot_emb.unsqueeze(0),
                                       decoder_output.unsqueeze(1).repeat(1, len(self.domain_slots), 1)), dim=-1)
                        )))
                        rel_scores = masked_softmax(rel_scores.squeeze(-1), mask_rel_last_y)
                    
                        # soft gate g_2
                        copy_rel_prob = torch.sigmoid(self.copy_rel_linear(context))
    
                        # slot gate
                        gate_output = self.slot_gate(context)
                        all_gate_output[domain_slot_index, turn_idx, :] = gate_output.squeeze(0)
                        gate = torch.argmax(gate_output.squeeze(0))
                        if gate == 1: # none
                            decoded_index = [self.none_idx, self.eos_idx]
                            break
                        elif gate == 2: # dontcare
                            decoded_index = [self.dontcare_idx, self.eos_idx]
                            break
    
                    # final distribution
                    final_dist = (1 - copy_rel_prob) * gen_prob * gen_dist
                    final_dist.scatter_add_(1, turn, (1 - copy_rel_prob) * (1 - gen_prob) * scores)
                    final_dist.scatter_add_(1, last_y[:, :, decode_idx].transpose(0, 1), copy_rel_prob * rel_scores)
    
                    # next input
                    top_idx = torch.argmax(final_dist, dim=1)
                    decoded_index.append(top_idx)
                    decoder_input = self.embedding(top_idx.unsqueeze(-1))
    
                # update
                tmp_y = []
                for di in decoded_index:
                    tmp_y.append(di)
                    if di == self.eos_idx:
                        break
                new_y.append(tmp_y)
    
            # update last y
            last_y, _ = self.pad_y([new_y], max_length=max_decode_length)
            all_decoded_index[:, turn_idx, :] = last_y[:, 0, :]
    
        return all_decoded_index, all_gate_output
