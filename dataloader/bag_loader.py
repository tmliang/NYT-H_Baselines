import os
import json
import pickle
import torch.utils.data
from collections import Counter
from .base_loader import BaseDataset


class BagDataset(BaseDataset):
    def __init__(self, file_name, opt):
        super().__init__(file_name, opt)
        self.training = True if 'train' in file_name else False
        self.data = self.load_data()
        print("Bag num of %s set: %d" % (file_name, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        bag = list(zip(*self.data[idx]))
        word, pos1, pos2, mask, length, rel, label = [torch.tensor(x, dtype=torch.long) for x in bag]
        hot_label = torch.zeros(self.rel_num, dtype=torch.long)
        if self.training:
            hot_rel = rel[0]
        else:
            if self.label2id['yes'] in label:
                true_rel = set((rel * label).numpy())
                true_rel.discard(0)
                hot_label[list(true_rel)] = 1
            hot_rel = torch.zeros(self.rel_num, dtype=torch.long)
            hot_rel[torch.unique(rel)] = 1
        return word, pos1, pos2, mask, length, hot_rel, hot_label

    def load_data(self):
        save_dir = os.path.join(self.processed_data_dir, self.file_name + '_bag.pkl')
        try:
            data = pickle.load(open(save_dir, 'rb'))
        except FileNotFoundError:
            print("Processed data does not exist, pre-processing data...")
            groups = {}
            with open(self.data_dir, encoding='utf-8') as f:
                for line in f.readlines():
                    ins = json.loads(line)
                    bag_id = ins['head']['guid'] + ins['tail']['guid'] + str(self.rel2id[ins['relation']]) if self.training \
                        else ins['head']['guid'] + ins['tail']['guid']
                    if bag_id in groups:
                        groups[bag_id].append(self.get_ins(ins))
                    else:
                        groups[bag_id] = [self.get_ins(ins)]
            data = list(groups.values())
            pickle.dump(data, open(save_dir, 'wb'))
        return data

    def loss_weight(self):
        rel = []
        for bag in self.data:
            rel.append(bag[0][5])
        stat = Counter(rel)
        class_weight = torch.ones(self.rel_num, dtype=torch.float32)
        for k, v in stat.items():
            class_weight[k] = 1. / v**0.05
        if torch.cuda.is_available():
            class_weight = class_weight.cuda()
        return class_weight


def collate_fn(X):
    X = list(zip(*X))
    word, pos1, pos2, mask, length, rel, label = X
    scope = []
    ind = 0
    for bag in word:
        scope.append((ind, ind + bag.shape[0]))
        ind += bag.shape[0]
    scope = torch.tensor(scope, dtype=torch.long)
    rel = torch.stack(rel)
    label = torch.stack(label)
    word = torch.cat(word, 0)
    pos1 = torch.cat(pos1, 0)
    pos2 = torch.cat(pos2, 0)
    mask = torch.cat(mask, 0)
    length = torch.cat(length, 0)
    return word, pos1, pos2, mask, length, rel, label, scope


class BagLoader(torch.utils.data.DataLoader):
    def __init__(self, filename, opt, shuffle=False):
        dataset = BagDataset(filename, opt)
        super().__init__(dataset=dataset,
                         batch_size=opt['batch_size'],
                         shuffle=shuffle,
                         pin_memory=True,
                         num_workers=opt['num_workers'],
                         collate_fn=collate_fn)
        self.word_vec_dir = dataset.word_vec_dir
        self.rel_num = dataset.rel_num
