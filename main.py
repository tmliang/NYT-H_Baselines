import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import numpy as np
from torch import optim
from Net.Model import Model
from sklearn import metrics
from dataloader import BagLoader
from config import config
from utils import AverageMeter, setup_seed, partial_acc


def train(train_loader, test_loader, opt):
    model = Model(train_loader.word_vec_dir, train_loader.rel_num, opt)
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss(weight=train_loader.dataset.loss_weight())
    optimizer = optim.SGD(model.parameters(), lr=opt['lr'], weight_decay=opt['weight_decay'])
    step_num = len(train_loader)
    not_best_count = 0
    best_result = 0
    ckpt = os.path.join(opt['output_dir'], opt['name']+'.pth.tar')
    for epoch in range(opt['epoch']):
        print("\n------------------- Epoch %d -------------------" % epoch)
        model.train()
        avg_loss = AverageMeter()
        avg_pos_acc = AverageMeter()
        avg_neg_acc = AverageMeter()
        for i, data in enumerate(train_loader):
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda()
            word, pos1, pos2, mask, length, rel, _, scope = data
            output = model(word, pos1, pos2, mask, length, scope, rel)
            loss = criterion(output, rel)
            _, pred = torch.max(output, -1)
            pos_acc = partial_acc(pred, rel, rel > 0)
            neg_acc = partial_acc(pred, rel, rel == 0)
            # Log
            avg_loss.update(loss.item(), 1)
            avg_pos_acc.update(pos_acc, 1)
            avg_neg_acc.update(neg_acc, 1)
            sys.stdout.write('\r[step: %d/%d] loss: %f, pos_acc: %f, neg acc: %f' %
                             (i+1, step_num, avg_loss.avg, avg_pos_acc.avg, avg_neg_acc.avg))
            sys.stdout.flush()
            # Optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("")
        if avg_pos_acc.avg > 0.5:
            result = test(test_loader, model)
            if result > best_result:
                print("Best result!")
                best_result = result
                torch.save({'state_dict': model.state_dict()}, ckpt)
                not_best_count = 0
            else:
                not_best_count += 1
            if not_best_count >= opt['early_stop']:
                break


def test(test_loader, model):
    model.eval()
    rel_num = test_loader.rel_num
    step_num = len(test_loader)
    all_rel = []
    all_pred = []
    all_label = []
    avg_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda()
            word, pos1, pos2, mask, length, rel, label, scope = data
            output = model(word, pos1, pos2, mask, length, scope)
            rel_log = rel.argmax(1)
            pred_log = torch.argmax(output, dim=1)
            pred = F.one_hot(pred_log, num_classes=rel_num)
            all_rel.append(rel)
            all_pred.append(pred)
            all_label.append(label)
            # Log
            acc = partial_acc(pred_log, rel_log, rel_log > 0)
            avg_acc.update(acc, 1)
            sys.stdout.write('\r[step: %d/%d] acc: %f' % (i+1, step_num, avg_acc.avg))
            sys.stdout.flush()
    print("")
    all_rel = torch.cat(all_rel, 0)
    all_pred = torch.cat(all_pred, 0)
    all_label = torch.cat(all_label, 0)

    # grouping bags of multi-labels
    ind = all_label.sum(1) > 1
    label_view = all_label[ind]
    pred_view = all_pred[ind]
    ind_false = (label_view * pred_view).sum(1) == 0
    res_label = torch.argmax(label_view[ind_false], dim=1, keepdim=True)
    label_view[ind_false] = torch.zeros(ind_false.sum(), rel_num, dtype=torch.long, device=label_view.device).scatter_(1, res_label, 1)
    label_view[~ind_false] = pred_view[~ind_false]
    all_label[ind] = label_view
    assert (all_label.sum(1) > 1).sum().item() == 0

    # evaluating
    all_rel = all_rel.detach().cpu().numpy()
    all_pred = all_pred.detach().cpu().numpy()
    all_label = all_label.detach().cpu().numpy()
    results = np.zeros([rel_num - 1, 3], dtype=np.float32)
    for r in range(1, rel_num):
        ind = all_rel[:, r] == 1
        label = all_label[:, r]
        pred = all_pred[:, r]
        results[r-1, 0] = metrics.precision_score(label[ind], pred[ind], zero_division=0)
        results[r-1, 1] = metrics.recall_score(label[ind], pred[ind])
        results[r-1, 2] = metrics.f1_score(label[ind], pred[ind])
    result = results.mean(0)
    print("[test result] precision: %f, recall: %f, f1-score: %f" % (result[0], result[1], result[2]))
    return result[2]


if __name__ == '__main__':
    opt = config()
    setup_seed(opt['seed'])
    train_loader = BagLoader('train', opt, shuffle=True)
    test_loader = BagLoader('test', opt, shuffle=False)
    train(train_loader, test_loader, opt)
