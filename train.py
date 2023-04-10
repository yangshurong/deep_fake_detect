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
                        default='./configs/caddm_train.cfg')
    parser.add_argument('--ckpt', type=str, help='The checkpoint of the pretrained model.',
                        default='./checkpoints/wadwd')
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

        outputs = net(batch_data)
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


def train():
    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = model.get(cfg)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    # optimizer init.
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

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

    # get training data
    print(cfg)
    train_dataset = DeepfakeDatasetSBI('train', cfg)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=cfg['num_worker'],
                              collate_fn=my_collate, pin_memory=True
                              )
    test_loaders = {}
    for k, v in cfg['test']['dataset'].items():
        test_dataset = DeepfakeDatasetTEST(k, v, cfg)

        test_loader = DataLoader(test_dataset,
                                 batch_size=cfg['test']['batch_size'],
                                 shuffle=True, num_workers=cfg['num_worker'],
                                 pin_memory=True
                                 )
        test_loaders[k] = test_loader
    # start trining.
    time_out = 0
    time_enter = 0
    for epoch in range(base_epoch, cfg['train']['epoch_num']):
        net.train()
        for index, (batch_data, batch_labels) in enumerate(train_loader):
            time_enter = int(time.time())
            if cfg['log_time_cost'] and index != 0:
                LOGGER.info(f'dataload one cost{time_enter-time_out}')
            lr = update_learning_rate(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            labels, location_labels, confidence_labels = batch_labels
            labels = labels.long().to(device)
            location_labels = location_labels.to(device)
            confidence_labels = confidence_labels.long().to(device)
            batch_data = batch_data.to(device)

            optimizer.zero_grad()
            locations, confidence, outputs = net(batch_data)
            # LOGGER.info()
            loss_end_cls = criterion(outputs, labels)
            loss_l, loss_c = det_criterion(
                (locations, confidence),
                confidence_labels, location_labels
            )
            acc = sum(outputs.max(-1).indices ==
                      labels).item() / labels.shape[0]
            det_loss = 0.1 * (loss_l + loss_c)
            loss = det_loss + loss_end_cls
            loss.backward()

            torch.nn.utils.clip_grad_value_(net.parameters(), 2)
            optimizer.step()

            outputs = [
                "e:{},iter: {}".format(epoch, index),
                "acc: {:.6f}".format(acc),
                "loss: {:.8f} ".format(loss.item()),
                "lr:{:.4g}".format(lr),
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
                with torch.no_grad():
                    test_one_epoch(k, net, device, v, cfg, optimizer, epoch)


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
