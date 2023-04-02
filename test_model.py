from backbones.caddm import CADDM
from backbones.cross_efficient_vit import CrossEfficientViT
from backbones.mcx_api import API_Net
import torch
from lib.util import load_config
import yaml


def test_CADDM():
    net = CADDM(2, 'resnet34').cuda()
    net.train()
    x = torch.rand((5, 3, 224, 224)).cuda()
    y = net(x)
    print(y.shape)


def test_cross_vit():

    cfg = load_config('./configs/cross_efficient_vit.cfg')
    model = CrossEfficientViT(config=cfg).cuda()
    x = torch.randn(5, 3, 224, 224).cuda()
    y, z, k = model(x)
    print(y.shape)


def test_MCX():
    cfg = load_config('./configs/mcx_api.cfg')
    model = API_Net(cfg).cuda()
    state_dict = torch.load('./network/model_mcx-api-rgb.tar')['state_dict']
    res = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '')
        res[k] = v
    del res['fc.weight']
    del res['fc.bias']
    x = torch.randn(5, 3, 448, 448).cuda()
    model.load_state_dict(res, strict=False)

    torch.save({
        'network': res
    }, './network/mcx_api_init.pkl')
    logit1_self, logit1_other, logit2_self, logit2_other, labels1, labels2, features = model(
        x, torch.tensor([1, 1, 1, 1, 1]))


if __name__ == '__main__':
    # test_CADDM()
    test_MCX()
    pass
