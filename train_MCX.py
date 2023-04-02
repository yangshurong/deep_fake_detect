#!/usr/bin/env python3
import time
from dataset import DeepfakeDatasetSBI, DeepfakeDatasetFF, DeepfakeDatasetTEST
import argparse
from collections import OrderedDict
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from lib.util import get_video_auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

import model
from detection_layers.modules import MultiBoxLoss
from trainlog import get_logger
from lib.util import load_config, update_learning_rate, my_collate
LOGGER = get_logger()


def args_func():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='The path to the config.',
                        default='./configs/mcx_api.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.',
                        default='./checkpoints/mcx_api.pth')
    args = parser.parse_args()
    return args


def save_checkpoint(net, opt, save_path, epoch_num, model_name, is_best=False):
    os.makedirs(save_path, exist_ok=True)
    checkpoint = {
        'network': net.state_dict(),
        'opt_state': opt.state_dict(),
        'epoch': epoch_num,
    }
    if is_best:
        torch.save(checkpoint, f'{save_path}/{model_name}_best.pkl')
        LOGGER.info(f'success save {save_path}/{model_name}_best.pkl')
    else:
        torch.save(
            checkpoint, f'{save_path}/{model_name}_epoch_{epoch_num}.pkl')
        LOGGER.info(
            f'success save {save_path}/{model_name}_epoch_{epoch_num}.pkl')


def load_checkpoint(ckpt, net, opt):
    base_epoch = 0
    if not os.path.exists(ckpt):
        return net, opt, base_epoch
    checkpoint = torch.load(ckpt)

    if 'network' in checkpoint:
        net.load_state_dict(checkpoint['network'], strict=False)
    if 'opt_state' in checkpoint:
        if len(checkpoint['opt_state']) != 0:
            opt.load_state_dict(checkpoint['opt_state'])

    if 'epoch' in checkpoint:
        base_epoch = int(checkpoint['epoch'])+1
    LOGGER.info(f'finish load {ckpt}')    
    return net, opt, base_epoch


# DATASET = {
#     "FF++": DeepfakeDatasetFF,
#     # 'SBI': DeepfakeDatasetSBI
#     'SBI': DeepfakeDatasetSBILMDB
# }
AUC_MAX = {}


def test_one_epoch(set_name, net, device, test_loader, cfg, optimizer, epoch):
    frame_pred_list = list()
    frame_label_list = list()
    video_name_list = list()
    sum_acc = 0
    for index, (batch_data, batch_labels) in enumerate(test_loader):

        labels, video_name = batch_labels
        labels = labels.long().to(device)
        batch_data = batch_data.to(device)

        # outputs = net(batch_data)
        outputs = net(batch_data, targets=None, flag='val')
        acc = sum(outputs.max(-1).indices ==
                  labels).item() / labels.shape[0]
        outputs = outputs[:, 1]
        sum_acc += acc
        frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
        frame_label_list.extend(labels.detach().cpu().numpy().tolist())
        video_name_list.extend(list(video_name))
        if index % cfg['log_interval'] == 0:
            LOGGER.info(f'finish eval {index} acc is {acc:.4f}')

    roc_auc = roc_auc_score(frame_label_list, frame_pred_list)
    LOGGER.info(
        f"{set_name} for roc_auc is {roc_auc:.4f} acc is {sum_acc/len(test_loader):.4f}")
    if set_name not in AUC_MAX:
        AUC_MAX[set_name] = 0.0

    if roc_auc > AUC_MAX[set_name]:
        AUC_MAX[set_name] = roc_auc

    LOGGER.info(f"{set_name} for best roc_auc is {AUC_MAX[set_name]:.4f}")


class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        # print(f'features.shape {features.shape}')
        # print(f'labels.shape {labels.shape}')
        device = (torch.device('cuda')
                  if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / \
            (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss


def train():
    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = model.get(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(cfg)
    train_dataset = DeepfakeDatasetSBI('train', cfg)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=cfg['num_worker'],
                              collate_fn=my_collate, pin_memory=True,drop_last=True
                              )
    test_loaders = {}
    for k, v in cfg['test']['dataset'].items():
        test_dataset = DeepfakeDatasetTEST(k, v, cfg)

        test_loader = DataLoader(test_dataset,
                                 batch_size=cfg['test']['batch_size'],
                                 shuffle=False, num_workers=cfg['num_worker'],
                                 pin_memory=True,
                                 drop_last=True
                                 )
        test_loaders[k] = test_loader

    net = net.to(device)
    # optimizer init.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01,
                                momentum=0.9,
                                weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 100*len(train_loader))
    # load checkpoint if given
    base_epoch = 0
    if args.ckpt:
        net, optimizer, base_epoch = load_checkpoint(args.ckpt, net, optimizer)
        
    # net = nn.DataParallel(net)

    # loss init
    det_criterion = MultiBoxLoss(
        cfg['det_loss']['num_classes'],
        cfg['det_loss']['overlap_thresh'],
        cfg['det_loss']['prior_for_matching'],
        cfg['det_loss']['bkg_label'],
        cfg['det_loss']['neg_mining'],
        cfg['det_loss']['neg_pos'],
        cfg['det_loss']['neg_overlap'],
        cfg['det_loss']['encode_target'],
        cfg['det_loss']['use_gpu']
    )
    criterion = nn.CrossEntropyLoss()

    # start trining.
    time_out = 0
    time_enter = 0
    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    op_loss = OrthogonalProjectionLoss(gamma=0.5)
    op_lambda = 0.4
    softmax_layer = nn.Softmax(dim=1).to(device)

    for epoch in range(base_epoch, cfg['train']['epoch_num']):
        net.train()
        for index, (batch_data, batch_labels) in enumerate(train_loader):
            time_enter = int(time.time())
            if cfg['log_time_cost'] and index != 0:
                LOGGER.info(f'dataload one cost{time_enter-time_out}')

            # detect--------------------------------
            labels, location_labels, confidence_labels = batch_labels
            labels = labels.long().to(device)
            location_labels = location_labels.to(device)
            confidence_labels = confidence_labels.long().to(device)
            batch_data = batch_data.to(device)

            # MCX--------------------------------
            optimizer.zero_grad()
            logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, features = net(
                batch_data, labels, flag='train', dist_type='euclidean')
            
            # locations, confidence, logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, features = model(
            #     batch_data, labels, flag='train', dist_type='euclidean')
            batch_size = logit1_self.shape[0]
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)

            self_logits = torch.zeros(2*batch_size, 2).to(device)
            other_logits = torch.zeros(2*batch_size, 2).to(device)
            self_logits[:batch_size] = logit1_self
            self_logits[batch_size:] = logit2_self
            other_logits[:batch_size] = logit1_other
            other_logits[batch_size:] = logit2_other

            logits = torch.cat([self_logits, other_logits], dim=0)
            targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
            softmax_loss = criterion(logits, targets)

            self_scores = softmax_layer(self_logits)[torch.arange(2*batch_size).to(device).long(),
                                                     torch.cat([labels1, labels2], dim=0)]
            other_scores = softmax_layer(other_logits)[torch.arange(2*batch_size).to(device).long(),
                                                       torch.cat([labels1, labels2], dim=0)]
            flag = torch.ones([2*batch_size, ]).to(device)
            rank_loss = rank_criterion(self_scores, other_scores, flag)

            # orthogonal projection loss
            loss_op = op_loss(features, labels)

            # loss_l, loss_c = det_criterion(
            #     (locations, confidence),
            #     confidence_labels, location_labels
            # )
            # det_loss = 0.1 * (loss_l + loss_c)

            loss = softmax_loss + rank_loss + op_lambda * loss_op
            loss.backward()
            # measure accuracy and record loss
            acc = sum(logits.max(-1).indices ==
                      targets).item() / targets.shape[0]

            optimizer.step()
            lr_scheduler.step()

            outputs = [
                "e:{},iter: {}".format(epoch, index),
                "acc: {:.6f}".format(acc),
                "loss: {:.8f} ".format(loss.item()),
                "lr:{:.4g}".format(optimizer.state_dict()['param_groups'][0]['lr']),
            ]
            if index % cfg['log_interval'] == 0:
                LOGGER.info(" ".join(outputs))
            time_out = int(time.time())
        if (epoch+1) % cfg['save_epoch'] == 0:
            save_checkpoint(net, optimizer,
                            cfg['model']['save_path'],
                            epoch,
                            cfg['model']['name']
                            )
            net.eval()
            for k, v in test_loaders.items():
                test_one_epoch(k, net, device, v, cfg, optimizer, epoch)


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
