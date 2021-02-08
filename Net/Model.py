import torch
import torch.nn as nn
from torch.nn import functional as F
from Net import CNN, PCNN, BiGRU
import numpy as np


class Model(nn.Module):
    def __init__(self, word_dir, rel_num, opt, pos_dim=5, hidden_size=230):
        super(Model, self).__init__()
        vec = torch.from_numpy(np.load(word_dir))
        vec_dim = vec.shape[-1]
        unk = torch.randn(1, vec_dim)
        blk = torch.zeros(1, vec_dim)
        emb_dim = vec_dim + 2 * pos_dim
        self.word_embedding = nn.Embedding.from_pretrained(torch.cat([vec, unk, blk], 0), freeze=False, padding_idx=-1)
        self.encoder_name = opt['encoder']
        self.pos1_embedding = nn.Embedding(2 * opt['max_pos_length'] + 1, pos_dim)
        self.pos2_embedding = nn.Embedding(2 * opt['max_pos_length'] + 1, pos_dim)
        self.drop = nn.Dropout(opt['dropout'])

        if self.encoder_name == 'CNN':
            self.encoder = CNN(emb_dim)
            self.rel = nn.Linear(hidden_size, rel_num)

        elif self.encoder_name == 'BiGRU':
            self.encoder = BiGRU(emb_dim)
            self.rel = nn.Linear(hidden_size * 2, rel_num)

        else:
            self.encoder = PCNN(emb_dim)
            self.rel = nn.Linear(hidden_size * 3, rel_num)

        self.init_weight()

    def forward(self, X, pos1, pos2, mask, length, scope, relation=None):
        X = self.word_pos_embedding(X, pos1, pos2)
        if self.encoder_name == 'CNN':
            X = self.encoder(X)
        elif self.encoder_name == 'PCNN':
            X = self.encoder(X, mask)
        elif self.encoder_name == 'BiGRU':
            X = self.encoder(X, length)
        else:
            raise NotImplementedError
        X = self.drop(X)
        X = self.sentence_attention(X, scope, relation)
        return X

    def init_weight(self):
        nn.init.xavier_uniform_(self.pos1_embedding.weight)
        nn.init.xavier_uniform_(self.pos2_embedding.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        nn.init.zeros_(self.rel.bias)

    def word_pos_embedding(self, X, pos1, pos2):
        X = self.word_embedding(X)
        pos1 = self.pos1_embedding(pos1)
        pos2 = self.pos2_embedding(pos2)
        X = torch.cat([X, pos1, pos2], -1)
        return X

    def sentence_attention(self, X, scope, Rel=None):
        bag_output = []
        if Rel is not None:  # For training
            Rel = F.embedding(Rel, self.rel.weight)
            for i in range(scope.shape[0]):
                bag_rep = X[scope[i][0]: scope[i][1]]
                att_score = F.softmax(bag_rep.matmul(Rel[i]), 0).view(1, -1)  # (1, Bag_size)
                att_output = att_score.matmul(bag_rep)  # (1, dim)
                bag_output.append(att_output.squeeze())  # (dim, )
            bag_output = torch.stack(bag_output)
            bag_output = self.drop(bag_output)
            bag_output = self.rel(bag_output)
        else:  # For testing
            att_score = X.matmul(self.rel.weight.t())  # (Batch_size, dim) -> (Batch_size, R)
            for s in scope:
                bag_rep = X[s[0]:s[1]]  # (Bag_size, dim)
                bag_score = F.softmax(att_score[s[0]:s[1]], 0).t()  # (R, Bag_size)
                att_output = bag_score.matmul(bag_rep)  # (R, dim)
                bag_output.append(torch.diagonal(F.softmax(self.rel(att_output), -1)))
            bag_output = torch.stack(bag_output)
            # bag_output = F.softmax(bag_output, -1)
        return bag_output
