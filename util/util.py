import torch
import math
import json
import re
import numpy as np
import torch.nn.functional as F


def collate_wrapper(batch):
    x = [one[0] for one in batch]
    tl = [one[1] for one in batch]
    bs = [one[2] for one in batch]
    last_bs = [one[3] for one in batch]
    gate_y = [one[4] for one in batch]
    return x, tl, bs, last_bs, gate_y


def normalize_slot(slot):
    if 'leaveat' in slot:
        slot = slot.replace('leaveat', 'leave at')
    if 'arriveby' in slot:
        slot = slot.replace('arriveby', 'arrive by')
    if 'pricerange' in slot:
        slot = slot.replace('pricerange', 'price range')
    return slot


def masked_softmax(x, m=None, axis=-1):
    x = torch.clamp(x, min=-15.0, max=15.0)
    e_x = torch.exp(x - torch.max(x, dim=axis, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=axis, keepdim=True) + 1e-10)
    return softmax


def get_ontology(data_file):
    ontology = {}
    ignore_domains = ['bus', 'hospital']
    with open(data_file, 'r') as f:
        slot_values_dict = json.load(f)
    for slot, values in slot_values_dict.items():
        domain = slot.split('-')[0]
        if domain in ignore_domains:
            continue
        ontology[slot] = values
    return ontology


"""
copyright 2019-present https://jasonwu0731.github.io/

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


def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text


def normalize(text):
    # lower case every word
    text = text.lower()
    text = text.split('|')[0]

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text) # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)

    fin = open('../../util/mapping.pair', 'r', encoding='utf-8')
    replacements = []
    for line in fin.readlines():
        tok_from, tok_to = line.replace('\n', '').split('\t')
        replacements.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))
    fin.close()

    for fromx, tox in replacements:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text


def masked_cross_entropy(logits, targets, masks, device):
    flat_logits = logits.view(-1, logits.size(-1))
    flat_logits = torch.log(flat_logits + 1e-10)
    flat_targets = targets.view(-1, 1)
    loss = -torch.gather(flat_logits, dim=1, index=flat_targets)
    loss = loss.view(*targets.size())
    loss = masking(loss, masks, device)
    return loss


def masking(loss, mask, device):
    mask_ = []
    batch_size = mask.size(0)
    max_len = loss.size(2)
    for si in range(mask.size(1)):
        seq_range = torch.arange(0, max_len).long().to(device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_range_expand = seq_range_expand
        seq_length_expand = mask[:, si].unsqueeze(1).expand_as(seq_range_expand)
        mask_.append((seq_range_expand < seq_length_expand))
    mask_ = torch.stack(mask_)
    mask_ = mask_.transpose(0, 1)
    loss = loss * mask_.float()
    loss = loss.sum() / (mask_.sum().float())
    return loss


def attend(seq, query, lens):
    scores = query.unsqueeze(1).expand_as(seq).mul(seq).sum(2)

    max_len = max(lens)
    for i, l in enumerate(lens):
        if l < max_len:
            scores.data[i, l:] = -np.inf
    scores = F.softmax(scores, dim=1)
    context = scores.unsqueeze(2).expand_as(seq).mul(seq).sum(1)
    return context, scores
