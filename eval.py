import torch
import os
import random
import numpy as np
from sklearn import metrics
from data_loader import data_loader
from Net.Model import Model
from config import config
from utils import AverageMeter


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def valid(test_loader, model):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                data = [x.cuda() for x in data]
            word, pos1, pos2, ent1, ent2, mask, length, scope, rel = data
            output = model(word, pos1, pos2, mask, length, scope)
            # Log
            y_true.append(rel[:, 1:])
            y_pred.append(output[:, 1:])
    y_true = torch.cat(y_true).reshape(-1).detach().cpu().numpy()
    y_pred = torch.cat(y_pred).reshape(-1).detach().cpu().numpy()
    return y_true, y_pred


def test(test_loader, pn_loaders, opt):
    ckpt = opt['ckpt']
    model = Model(test_loader.dataset.vec_save_dir, test_loader.dataset.rel_num(), opt)
    if torch.cuda.is_available():
        model = model.cuda()
    state_dict = torch.load(ckpt)['state_dict']
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        own_state[name].copy_(param)
    print("=== Denoising Evaluation ===")
    y_true, y_pred = valid(test_loader, model)
    _, pr = compute_curve(y_true, y_pred)
    for i, pn_loader in enumerate(pn_loaders):
        if i == 0:
            mode = 'one'
        elif i == 1:
            mode = 'two'
        else:
            mode = 'all'
        print("pn_mode:", mode)
        y_true_pn, y_pred_pn = valid(pn_loader, model)
        compute_pn(y_true_pn, y_pred_pn)
    curve_dir = opt['curve_dir']
    if not os.path.exists(curve_dir):
        os.makedirs(curve_dir)
    np.save(os.path.join(curve_dir, 'precision.npy'), pr[0])
    np.save(os.path.join(curve_dir, 'recall.npy'), pr[1])


def compute_curve(y_true, y_pred):
    order = np.argsort(-y_pred)
    correct = 0.
    total = y_true.sum()
    precision = []
    recall = []
    for i, o in enumerate(order):
        correct += y_true[o]
        precision.append(float(correct) / (i + 1))
        recall.append(float(correct) / total)
    precision = np.array(precision)
    recall = np.array(recall)
    auc = metrics.average_precision_score(y_true, y_pred)
    print("auc: {0:.3f}".format(auc*100))
    return auc, [precision, recall]


def compute_pn(y_true, y_pred):
    order = np.argsort(-y_pred)
    p100 = (y_true[order[:100]]).mean()*100
    p200 = (y_true[order[:200]]).mean()*100
    p300 = (y_true[order[:300]]).mean()*100
    mean = (p100+p200+p300)/3
    print("P@100: {0:.1f}, P@200: {1:.1f}, P@300: {2:.1f}, Mean: {3:.1f}".format(p100, p200, p300, mean))


if __name__ == '__main__':
    opt = vars(config())
    test_loader = data_loader(opt['test'], opt, shuffle=False, training=False)
    pn_1_loader = data_loader(opt['test'], opt, shuffle=False, training=False, pn_mode='one')
    pn_2_loader = data_loader(opt['test'], opt, shuffle=False, training=False, pn_mode='two')
    pn_all_loader = data_loader(opt['test'], opt, shuffle=False, training=False, pn_mode='all')
    pn_loaders = [pn_1_loader, pn_2_loader, pn_all_loader]
    test(test_loader, pn_loaders, opt)

