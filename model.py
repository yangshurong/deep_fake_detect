#!/usr/bin/env python3
import torch
from backbones.caddm import CADDM
from backbones.cross_efficient_vit import CrossEfficientViT
from backbones.mcx_api import API_Net

def get(cfg):
    """
    load one model
    :param model_path: ./models
    :param model_type: source/target/det
    :param model_backbone: res18/res34/Efficient
    :param use_cuda: True/False
    :return: model
    """
    if cfg['model']['name'] == 'caddm':
        model = CADDM(2, backbone=cfg['model']['backbone'])
    elif cfg['model']['name'] == 'cross_vit_caddm':
        model = CrossEfficientViT(cfg)
    elif cfg['model']['name'] == 'mcx_api':
        model = API_Net(cfg)
    return model


if __name__ == "__main__":
    m = get()
# vim: ts=4 sw=4 sts=4 expandtab
