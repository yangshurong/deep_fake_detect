from backbones.caddm import CADDM
from backbones.cross_efficient_vit import CrossEfficientViT
from backbones.mcx_api import API_Net
from backbones.class_layer.inceptionnext import inceptionnext_small
import torch
import torch.nn as nn
from lib.util import load_config
import yaml
from torchvision import models
from torchstat import stat


def test_CADDM():
    # net = CADDM(2, 'resnet34').cuda()
    x = torch.rand((5, 3, 224, 224)).cuda()
    # y,global_feat=net.base_model(x)
    # net.train()

    # y = net(x)
    net = inceptionnext_small(True).cuda()
    y, global_feats = net(x)
    print(y.shape)
    print(global_feats.shape)


def test_cross_vit():
    print(torch.cuda.is_available())
    cfg = load_config('./configs/cross_efficient_vit.cfg')
    model = CrossEfficientViT(config=cfg)
    stat(model, (3, 224, 224))
    # x = torch.randn(5, 3, 224, 224).cuda()
    # y, z, k = model(x)
    # print(y.shape)


def test_MCX():
    cfg = load_config('./configs/mcx_api.cfg')
    model = API_Net(cfg).cuda()
    x = torch.randn(5, 3, 448, 448).cuda()
    # model = models.resnet101(pretrained=True).cuda()
    # model = inceptionnext_small(True).cuda()
    # layers = list(model.children())
    # conv = nn.Sequential(*layers)
    # y = conv(x)
    # print(y.shape)
    logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, features = model(
        x, torch.tensor([1, 1, 1, 1, 1]))


if __name__ == '__main__':
    test_MCX()
    pass
