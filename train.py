#!/usr/bin/env python3
from dataset import DeepfakeDatasetSBI, DeepfakeDatasetDFDC, DeepfakeDatasetFF
import argparse
from collections import OrderedDict
import os
from sklearn.metrics import roc_auc_score
from lib.util import get_video_auc

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
                        default='./checkpoints/resnet34.pkl')
    args = parser.parse_args()
    return args


def save_checkpoint(net, opt, save_path, epoch_num):
    os.makedirs(save_path, exist_ok=True)
    net.to('cpu')

    checkpoint = {
        'network': net.state_dict(),
        'opt_state': opt.state_dict(),
        'epoch': epoch_num,
    }

    torch.save(checkpoint, f'{save_path}/epoch_{epoch_num}.pkl')
    net.to('cuda:0')


def load_checkpoint(ckpt, net, opt):
    checkpoint = torch.load(ckpt)

    # gpu_state_dict = OrderedDict()
    # for k, v in checkpoint['network'] .items():
    #     name = "module."+k  # add `module.` prefix
    #     gpu_state_dict[name] = v.to(device)
    net.load_state_dict(checkpoint['network'])
    # opt.load_state_dict(checkpoint['opt_state'])
    base_epoch = int(checkpoint['epoch']) + 1
    return net, opt, base_epoch


DATASET = {
    "FF++": DeepfakeDatasetFF,
    'DFDC': DeepfakeDatasetDFDC,
    'SBI': DeepfakeDatasetSBI
}


def train():
    args = args_func()

    # load conifigs
    cfg = load_config(args.cfg)

    # init model.
    net = model.get(backbone=cfg['model']['backbone'])
    # optimizer init.
    optimizer = optim.AdamW(net.parameters(), lr=1e-3, weight_decay=4e-3)

    # load checkpoint if given
    base_epoch = 0
    if args.ckpt:
        net, optimizer, base_epoch = load_checkpoint(args.ckpt, net, optimizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
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
    train_dataset = DATASET[cfg['train']['name']]('train', cfg)
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['train']['batch_size'],
                              shuffle=True, num_workers=4,
                              collate_fn=my_collate
                              )

    test_dataset = DATASET[cfg['test']['name']]('test', cfg)
    test_loader = DataLoader(test_dataset,
                             batch_size=cfg['test']['batch_size'],
                             shuffle=True, num_workers=4,
                             )
    # start trining.

    for epoch in range(base_epoch, cfg['train']['epoch_num']):
        net.train()
        for index, (batch_data, batch_labels) in enumerate(train_loader):

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
                "acc: {:.2f}".format(acc),
                "loss: {:.8f} ".format(loss.item()),
                "lr:{:.4g}".format(lr),
            ]
            LOGGER.info(" ".join(outputs))
        save_checkpoint(net, optimizer,
                        cfg['model']['save_path'],
                        epoch)

        # start testing.
        frame_pred_list = list()
        frame_label_list = list()
        video_name_list = list()
        net.eval()
        for batch_data, batch_labels in test_loader:

            labels, video_name = batch_labels
            labels = labels.long().to(device)
            batch_data = batch_data.to(device)

            outputs = net(batch_data)
            outputs = outputs[:, 1]
            frame_pred_list.extend(outputs.detach().cpu().numpy().tolist())
            frame_label_list.extend(labels.detach().cpu().numpy().tolist())
            video_name_list.extend(list(video_name))

        f_auc = roc_auc_score(frame_label_list, frame_pred_list)
        v_auc = get_video_auc(
            frame_label_list, video_name_list, frame_pred_list)
        print(f"Frame-AUC of {cfg['test']['name']} is {f_auc:.4f}")
        print(f"Video-AUC of {cfg['test']['name']} is {v_auc:.4f}")


if __name__ == "__main__":
    train()

# vim: ts=4 sw=4 sts=4 expandtab
