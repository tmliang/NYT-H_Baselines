import os
import json
import numpy as np
import torch.utils.data


def init_dir(*dirs):
    for folder in dirs:
        if not os.path.exists(folder):
            os.makedirs(folder)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, file_name, opt):
        super().__init__()
        self.file_name = file_name
        self.max_length = opt['max_length']
        self.max_pos_length = opt['max_pos_length']
        self.data_dir = os.path.join(opt['data_dir'], file_name + '.json')
        self.processed_data_dir = opt['processed_data_dir']
        self.out_dir = os.path.join(opt['output_dir'], opt['name'])
        self.rel2id = json.load(open(os.path.join(opt['data_dir'], 'rel2id.json')))
        self.label2id = json.load(open(os.path.join(opt['data_dir'], 'bag_label2id.json')))
        init_dir(self.processed_data_dir, self.out_dir)
        self.word2id, self.word_vec_dir = self.init_word(opt['word_vec_dir'])
        self.rel_num = len(self.rel2id)

    def init_word(self, word_vec_dir):
        dim = word_vec_dir.split('.')[-2]
        word2id_dir = os.path.join(self.processed_data_dir, 'word2id.%s.json' % dim)
        vec_dir = os.path.join(self.processed_data_dir, 'word_vec.%s.npy' % dim)
        try:
            word2id = json.load(open(word2id_dir))
        except FileNotFoundError:
            word2id = {}
            word_vec = []
            with open(word_vec_dir, encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip().split()
                    word_vec.append(line[1:])
                    word2id[line[0].lower()] = len(word2id)
            word2id['[UNK]'] = len(word2id)
            word2id['[PAD]'] = len(word2id)
            np.save(vec_dir, np.array(word_vec, dtype=np.float32))
            json.dump(word2id, open(word2id_dir, 'w'))
        return word2id, vec_dir

    def pos_emb(self, word, sentence):
        p = sentence.find(' ' + word + ' ')
        if p == -1:
            p = len(sentence) - len(word) if sentence[-len(word) - 1:] == ' ' + word else 0
        else:
            p += 1
        p = sentence[:p].count(' ')
        pos = np.arange(self.max_length) - p + self.max_pos_length
        pos[pos > 2 * self.max_pos_length] = 2 * self.max_pos_length
        pos[pos < 0] = 0
        return pos, p

    def get_ins(self, ins):
        # word
        seq = ins['sentence'].lower().split()
        _word = np.zeros(self.max_length, dtype=np.long)
        for i, w in enumerate(seq):
            if i < self.max_length:
                _word[i] = self.word2id[w] if w in self.word2id else self.word2id['[UNK]']
            else:
                break
        seq_len = len(seq)
        _word[seq_len:] = self.word2id['[PAD]']

        # pos
        _pos1, p1 = self.pos_emb(ins['head']['word'], ins['sentence'])
        _pos2, p2 = self.pos_emb(ins['tail']['word'], ins['sentence'])

        # mask
        p1, p2 = sorted((p1, p2))
        _mask = np.zeros(self.max_length, dtype=np.long)
        _mask[p2 + 1: seq_len] = 3
        _mask[p1 + 1: p2 + 1] = 2
        _mask[:p1 + 1] = 1
        _mask[seq_len:] = 0

        # sentence length
        _length = min(seq_len, self.max_length)

        # rel
        _rel = self.rel2id[ins['relation']]

        # noise label
        _label = self.label2id[ins['bag_label']]

        return [_word, _pos1, _pos2, _mask, _length, _rel, _label]

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

