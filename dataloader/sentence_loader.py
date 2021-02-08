import os
import json
import pickle
import torch.utils.data
from .base_loader import BaseDataset


class SentenceDataset(BaseDataset):
    def __init__(self, file_name, opt):
        super().__init__(file_name, opt)
        self.data = self.load_data()
        print("Sentence num of %s set: %d" % (file_name, len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self):
        save_dir = os.path.join(self.processed_data_dir, self.file_name + '_sen.pkl')
        try:
            data = pickle.load(open(save_dir, 'rb'))
        except FileNotFoundError:
            print("Processed data does not exist, pre-processing data...")
            data = []
            with open(self.data_dir, encoding='utf-8') as f:
                for line in f.readlines():
                    ins = json.loads(line)
                    data.append(self.get_ins(ins))
            pickle.dump(data, open(save_dir, 'wb'))
        return data


class SentenceLoader(torch.utils.data.DataLoader):
    def __init__(self, filename, opt, shuffle=False):
        dataset = SentenceDataset(filename, opt)
        super().__init__(dataset=dataset,
                         batch_size=opt['batch_size'],
                         shuffle=shuffle,
                         pin_memory=True,
                         num_workers=opt['num_workers'])
        self.word_vec_dir = dataset.word_vec_dir
        self.rel_num = dataset.rel_num
