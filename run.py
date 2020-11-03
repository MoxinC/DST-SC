import argparse
import copy
import json
import numpy as np
import os
import random
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from model.model import Model
from util.Dataset import WOZDataSet
from util.util import collate_wrapper, masked_cross_entropy, masking, get_ontology


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--emb_size", dest="embedding_size", type=int, metavar='<int>', default=400)
    parser.add_argument("--hidden_size", dest="hidden_size", type=int, metavar='<int>', default=400)
    parser.add_argument("--rnn_layers", dest="rnn_layers", type=int, metavar='<int>', default=1)
    parser.add_argument("--dropout", dest="dropout", type=float, metavar='<float>', default=0.2)
    parser.add_argument("--pretrained", dest="pretrained", default='embedding/embedding_6521.txt')
    parser.add_argument('--char_emb_size', dest='char_embedding_size', type=int, default=100)
    parser.add_argument('--char_pretrained', dest='char_pretrained', type=str, default='embedding/charNgram.txt')
    parser.add_argument('--use_char_emb', dest='use_char_embedding', type=bool, default=True)
    parser.add_argument('--fix_emb', dest='fix_embedding', type=bool, default=False)
    parser.add_argument('--word_dropout', dest='word_dropout', type=float, default=0.2)
    parser.add_argument('--max_decode_length', dest='max_decode_length', type=int, default=10)
    parser.add_argument('--teacher', dest='use_teacher_forcing', type=bool, default=True)
    parser.add_argument('--teacher_ratio', dest='teacher_forcing_ratio', type=float, default=0.5)
    parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=100)
    parser.add_argument("--lr", dest='lr', type=float, metavar='<float>', default=0.001)
    parser.add_argument('--seed', type=int, default=1111)
    parser.add_argument('--early_stop', dest='early_stop', type=int, default=6)
    parser.add_argument('--gpu', dest='gpu', type=str, default='-1')
    parser.add_argument("--batch_size", dest="batch_size", type=int, metavar='<int>', default=5)
    parser.add_argument('--gas', dest='gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--dataset', dest='dataset', type=str, default='2.0')
    parser.add_argument('--is_test', dest='is_test', type=bool, default=False)

    args = parser.parse_args()

    args.ontology = 'dataset/multiwoz/ontology.json'
    args.vocab = 'dataset/multiwoz/' + args.dataset + '/vocab.txt'
    args.train = 'dataset/multiwoz/' + args.dataset + '/train_dials.json'
    args.dev = 'dataset/multiwoz/' + args.dataset + '/dev_dials.json'
    args.test = 'dataset/multiwoz/' + args.dataset + '/test_dials.json'
    args.gpus = [int(g) for g in args.gpu.split(',')]

    return args


NUM_GATES = 3
ARGS = get_args()


class DST:

    def __init__(self, rank):
        self.rank = rank
        self.args = copy.deepcopy(ARGS)

        # set seed
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False

        # set device
        device = 'cpu'
        if torch.cuda.is_available():
            if self.args.gpus[rank] < 0:
                print("WARNING: You have a CUDA device, you didnt use it.")
            else:
                torch.cuda.set_device(self.args.gpus[rank])
                print("Rank-{0}: using device {1} ".format(rank, torch.cuda.current_device()))
                torch.cuda.manual_seed(self.args.seed)
                device = 'cuda'
        self.args.device = torch.device(device)

        # model
        self.mdl = Model(self.args).to(self.args.device)

        # load data files
        if not self.args.is_test:
            self.train_loader = self.create_loader(self.args.train, self.args.batch_size, True, True)
            all_len = 0
            for one in self.train_loader.dataset.x:
                all_len += len(one)
            avg_len = all_len / len(self.train_loader.dataset.x)
            print('Rank-{0}: train data {1}, avg length {2}'.format(rank, len(self.train_loader.dataset), avg_len))

        if rank == 0:
            self.dev_loader = self.create_loader(self.args.dev, 1, False)
            self.test_loader = self.create_loader(self.args.test, 1, False)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.mdl.parameters()), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=1,
                                                              min_lr=0.0001, verbose=False)
        self.cross_entropy = nn.CrossEntropyLoss()

        # record best parameter on dev
        self.best_parameter = None
        self.best_dev_result = 0
        self.best_dev_epoch = -1
        self.is_early_stop = torch.Tensor([0]).to(self.args.device)
        self.joint_accuracy = torch.FloatTensor([0]).to(self.args.device)

        self.slot_value = get_ontology(self.args.ontology)
        self.slots = list(self.slot_value.keys())

    def save_model(self, joint_accuracy, is_dev=False):
        args_dict = vars(self.args)
        if is_dev:
            dir_name = 'params/multiwoz/dev/' + str(joint_accuracy)
        else:
            dir_name = 'params/multiwoz/test/' + str(joint_accuracy)

        while os.path.exists(dir_name):
            print(dir_name + ' has existed')
            dir_name += '_1'
            print('use another directory: ' + dir_name)
        self.save_path = dir_name

        os.makedirs(dir_name)
        torch.save(self.best_parameter, dir_name + '/params.pkl')
        with open(dir_name + '/hyper_parameter.txt', 'w', encoding='utf-8') as f:
            for key, value in args_dict.items():
                f.write(str(key) + ': ' + str(value) + '\n')

    def create_loader(self, file_name, batch_size, shuffle, is_training=False, data_type='all'):
        dataset = WOZDataSet(file_name, self.args, is_training, data_type, shuffle)
        loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, 
                                             collate_fn=collate_wrapper)
        return loader

    @torch.no_grad()
    def evaluate(self, loader, is_test=False, save=False):
        eos_idx = self.mdl.eos_idx
        none_idx = self.mdl.none_idx
        
        num_slots = len(self.slots)
        total_turns = 0
        num_correct = [0] * num_slots
        num_joint_correct = 0
        num_gates = 0
        num_gates_correct = 0

        # for saving result
        idx2word = {v: k for k, v in self.mdl.vocab.items()}
        all_prediction = {}
        all_dialogue_idx = []
        if is_test:
            fp = open(self.args.test, 'r', encoding='utf-8')
        else:
            fp = open(self.args.dev, 'r', encoding='utf-8')
        json_data = json.load(fp)
        for dial in json_data:
            dial_idx = dial['dialogue_idx']
            all_dialogue_idx.append(dial_idx)
        fp.close()

        dial_idx = -1
        for x, _, y, _, gate_y in loader:
            dial_idx += 1

            outputs, gates = self.mdl.test(x, gate_y)
            flat_y = sum(y, [])
            targets, _ = self.mdl.pad_y(flat_y, max_length=self.args.max_decode_length)
            flat_gate_y = sum(gate_y, [])

            outputs = outputs.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
            gates = gates.detach().cpu().numpy()

            num_turns = outputs[0].shape[0]
            total_turns += num_turns
            for i in range(num_turns):
                slot_result = []
                turn_pred, turn_label = [], []
                for j in range(num_slots):
                    num_gates += 1
                    gate_predict = np.argmax(gates[j][i])
                    gate_label = flat_gate_y[i][j]
                    if gate_predict == gate_label:
                        num_gates_correct += 1

                    predict = []
                    for v in outputs[j][i]:
                        if v == eos_idx:
                            break
                        predict.append(v)

                    label = []
                    for v in targets[j][i]:
                        if v == eos_idx:
                            break
                        label.append(v)

                    result = np.array_equal(predict, label)
                    num_correct[j] += int(result)
                    slot_result.append(result)

                    # for saving result
                    slot_key = self.slots[j]
                    if len(predict) > 0 and predict[0] != none_idx:
                        turn_pred.append(slot_key + '-' + ' '.join([idx2word[idx] for idx in predict]))
                    if len(label) > 0 and label[0] != none_idx:
                        turn_label.append(slot_key + '-' + ' '.join([idx2word[idx] for idx in label]))

                if np.all(slot_result):
                    num_joint_correct += 1

                # for saving result
                if all_dialogue_idx[dial_idx] not in all_prediction:
                    all_prediction[all_dialogue_idx[dial_idx]] = {}
                all_prediction[all_dialogue_idx[dial_idx]][i] = {'turn_belief': turn_label, 'pred_bs_ptr': turn_pred}

        if save:
            with open('all_prediction.json', 'w+', encoding='utf-8') as f:
                json.dump(all_prediction, f, ensure_ascii=False, indent=4)

        joint_accuracy = 100 * num_joint_correct / total_turns
        overall_accuracy = 100 * sum(num_correct) / (total_turns * num_slots)
        gate_accuracy = 100 * num_gates_correct / num_gates
        print("Joint Accuracy={0}, Overall Accuracy={1}, Gate Accuracy={2}".format(
            joint_accuracy, overall_accuracy, gate_accuracy))

        return joint_accuracy


    def train(self):
        print('Rank-{0}: start training'.format(self.rank))

        max_joint_accuracy = -1
        step = 0
        world_step = torch.LongTensor([step]).to(self.args.device)
        for epoch in range(1, self.args.epochs + 1):
            t0 = time.clock()
            losses = []

            for x, _, y, last_y, gate_y in self.train_loader:
                step += 1
                if step % self.args.gradient_accumulation_steps == 0:
                    world_step = torch.LongTensor([step]).to(self.args.device)
                    reduce_step(world_step)

                all_decoder_output, pad_y, y_length, all_gate_output = self.mdl(x, y, last_y)
                gate_y = torch.LongTensor(gate_y).to(self.args.device)

                loss = 0
                decode_loss = masked_cross_entropy(all_decoder_output, pad_y, y_length, self.args.device)
                loss += decode_loss
                gate_loss = self.cross_entropy(all_gate_output.transpose(0, 1).contiguous().view(-1, NUM_GATES),
                                               gate_y.contiguous().view(-1))
                loss += gate_loss
                
                if self.rank == 0:
                    print('\rDecode Loss {0}, Gate Loss {1}'.format(decode_loss, gate_loss), end='', flush=True)

                losses.append(loss.item())
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.mdl.parameters(), 10.0)

                if step % self.args.gradient_accumulation_steps == 0:
                    average_gradients(self.mdl)
                    self.optimizer.step()
                    self.mdl.zero_grad()

            t1 = time.clock()
            if self.rank == 0:
                print("\r\nepoch={0} loss={1} time={2}s".format(epoch, np.mean(losses), t1 - t0))

            if self.rank == 0:
                self.mdl.eval()
                joint_accuracy = self.evaluate(self.dev_loader)
                self.mdl.train()

                self.joint_accuracy[0] = joint_accuracy

                if joint_accuracy > max_joint_accuracy:
                    max_joint_accuracy = joint_accuracy
                    self.best_parameter = copy.deepcopy(self.mdl.state_dict())
                    self.best_dev_epoch = epoch
                    # self.save_model(max_joint_accuracy)
                else:
                    if self.best_dev_epoch != -1 and epoch - self.best_dev_epoch == self.args.early_stop:
                        self.is_early_stop[0] = 1

            dist.broadcast(self.joint_accuracy, 0)
            dist.broadcast(self.is_early_stop, 0)

            self.scheduler.step(self.joint_accuracy.item())
            if self.is_early_stop.item() == 1:
                print("Rank-{0}: early stop".format(rank))
                break

        if self.rank == 0:
            print("Ending training")
            print("Max joint accuracy on dev is {0}".format(max_joint_accuracy))
            self.test()

    def test(self):
        self.mdl.load_state_dict(self.best_parameter)
        self.mdl.eval()
        joint_accuracy = self.evaluate(self.test_loader)
        self.save_model(joint_accuracy)

    def predict(self, param):
        self.mdl.load_state_dict(torch.load(param, map_location='cpu'))
        self.mdl.eval()
        print('Predicting ...')
        self.evaluate(self.test_loader, is_test=True, save=True)


def run(rank):
    dst = DST(rank)
    dst.train()


def average_gradients(model):
    size = float(dist.get_world_size())
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= size


def reduce_step(step):
    dist.all_reduce(step, op=dist.ReduceOp.SUM)


def init_process(host, port, rank, fn, backend='nccl'):
    os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=len(ARGS.gpus))
    fn(rank)


if __name__ == '__main__':
    if not ARGS.is_test:
        host = '127.0.0.1'
        port = '29501'
        processes = []
        for rank in range(len(ARGS.gpus)):
            p = mp.Process(target=init_process, args=(host, port, rank, run))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        dst = DST(0)
        dst.predict('') # todo: specify checkpoint file path
