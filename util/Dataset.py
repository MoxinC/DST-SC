import json
import numpy as np
import torch.distributed as dist

from collections import defaultdict
from torch.utils.data import Dataset
from util.fix_label import fix_general_label_error
from util.util import normalize_slot, normalize, get_ontology


class WOZDataSet(Dataset):
    def transfer_label(self, bs):
        all_slots_labels = {}
        for slot in self.slots:
            all_slots_labels[slot] = bs[slot] if bs.get(slot, '') != '' else 'none'
            all_slots_labels[slot] += ' EOS'
        return list(all_slots_labels.values())

    def transfer_gate(self, bs):
        simplified_bs = defaultdict(str)
        for domain_slot, value in bs.items():
            simplified_bs[domain_slot] = value

        turn_gates = [0] * len(self.slots)
        for slot_idx, slot in enumerate(self.slots):
            value = simplified_bs[slot] if simplified_bs[slot] != '' else 'none'
            if value == 'none':
                turn_gates[slot_idx] = 1
            elif value == 'dontcare':
                turn_gates[slot_idx] = 2

        return turn_gates

    def sentence2index(self, sentences):
        index = []
        for sentence in sentences:
            index.append([self.word2index.get(word, self.word2index['UNK']) for word in sentence.split()])
        return index

    def label2index(self, labels):
        index = []
        num_slots = len(labels[0])
        for label in labels:
            slot_index = []
            for i in range(num_slots):
                slot_index.append([self.word2index.get(word, self.word2index['UNK']) for word in label[i].split()])
            index.append(slot_index)
        return index

    def load_vocab(self):
        words = ['PAD', 'UNK', 'SEP', 'EOS']
        with open(self.args.vocab, 'r', encoding='utf-8') as f:
            for line in f:
                words.append(line[:-1])
        self.word2index = {w: i for i, w in enumerate(words)}
        self.index2word = {i: w for i, w in enumerate(words)}

    def __init__(self, file_name, args, is_training=False, data_type='all', shuffle=False):
        self.args = args
        self.load_vocab()
        slot_values_dict = get_ontology(self.args.ontology)
        self.slots = list(slot_values_dict.keys())

        all_x, all_tl, all_bs, all_last_bs, all_gate_y =  [], [], [], [], []
        self.x, self.tl, self.bs, self.last_bs, self.gate_y  = [], [], [], [], []

        with open(file_name, 'r', encoding='utf-8') as f:
            contents = json.load(f)
            for dialogue in contents:
                turns_transcript = []
                turns_system_transcript = []
                turns_tl = []
                turns_bs = []
                turns_last_bs = []
                turns_gate = []
                turn_bs = ['none EOS'] * len(self.slots)
                
                for turn in dialogue['dialogue']:
                    # transcript
                    transcript = turn['transcript']
                    turns_transcript.append(transcript)
                    # system transcript
                    system_transcript = turn['system_transcript']
                    turns_system_transcript.append(system_transcript)
                    # turn bs
                    bs = turn['belief_state']
                    bs = fix_general_label_error(bs, False, self.slots)
                    turn_last_bs = turn_bs
                    turn_bs = self.transfer_label(bs)
                    turns_bs.append(turn_bs)
                    turns_last_bs.append(turn_last_bs)
                    # turn label
                    tmp_tl = turn['turn_label']
                    tl = [{'slots': [l]} for l in tmp_tl]
                    tl = fix_general_label_error(tl, False, self.slots)
                    turn_tl = self.transfer_label(tl)
                    turns_tl.append(turn_tl)
                    # slot gate
                    turn_gate = self.transfer_gate(bs)
                    turns_gate.append(turn_gate)

                turns = [turns_system_transcript[i] + ' SEP ' + turns_transcript[i] for i in range(len(turns_transcript))]

                # dialogue history
                contexts = []
                turns_mask = []
                for turn_idx in range(len(turns)):
                    context = turns[:turn_idx + 1]
                    joint_context = ' SEP '.join(context)
                    contexts.append(joint_context)
                    turn_mask = [0] * len(joint_context.split())
                    current_turn_length = len(context[-1].split())
                    turn_mask[-current_turn_length:] = [1] * current_turn_length
                    turns_mask.append(turn_mask)

                if is_training:
                    all_x += self.sentence2index(contexts)
                    all_tl += self.label2index(turns_tl)
                    all_bs += self.label2index(turns_bs)
                    all_last_bs += self.label2index(turns_last_bs)
                    all_gate_y += turns_gate
                else:
                    self.x.append(self.sentence2index(contexts))
                    self.tl.append(self.label2index(turns_tl))
                    self.bs.append(self.label2index(turns_bs))
                    self.last_bs.append(self.label2index(turns_last_bs))
                    self.gate_y.append(turns_gate)

        if is_training:
            world_size = dist.get_world_size()
            rank = dist.get_rank()

            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(len(all_x)))
                all_x = np.array(all_x)[shuffle_indices]
                all_tl = np.array(all_tl)[shuffle_indices]
                all_bs = np.array(all_bs)[shuffle_indices]
                all_last_bs = np.array(all_last_bs)[shuffle_indices]
                all_gate_y = np.array(all_gate_y)[shuffle_indices]

            for i in range(len(all_x)):
                if i % world_size == rank:
                    self.x.append(all_x[i])
                    self.tl.append(all_tl[i])
                    self.bs.append(all_bs[i])
                    self.last_bs.append(all_last_bs[i])
                    self.gate_y.append(all_gate_y[i])

    def __getitem__(self, index):
        return self.x[index], self.tl[index], self.bs[index], self.last_bs[index], self.gate_y[index]

    def __len__(self):
        return len(self.x)
