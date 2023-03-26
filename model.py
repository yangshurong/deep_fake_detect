#!/usr/bin/env python3
import torch
from backbones.caddm import CADDM
from backbones.cross_efficient_vit import CrossEfficientViT


def get(pretrained_model=None, backbone='efficientnet-b4', cfg=None):
    """
    load one model
    :param model_path: ./models
    :param model_type: source/target/det
    :param model_backbone: res18/res34/Efficient
    :param use_cuda: True/False
    :return: model
    """
    if backbone not in ['resnet34', 'efficientnet-b3', 'efficientnet-b4', 'cross_efficient_vit']:
        raise ValueError("Unsupported type of models!")
    if backbone in ['resnet34', 'efficientnet-b3', 'efficientnet-b4']:
        model = CADDM(2, backbone=backbone)
    else:
        model = CrossEfficientViT(cfg)
    if pretrained_model:
        checkpoint = torch.load(pretrained_model)
        model.load_state_dict(checkpoint['network'])
    return model


if __name__ == "__main__":
    m = get()
# vim: ts=4 sw=4 sts=4 expandtab
