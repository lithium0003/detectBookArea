from torchvision.models import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, EfficientNet
from torchvision.models.efficientnet import Conv2dNormActivation
from torchvision.models.efficientnet import EfficientNet, FusedMBConvConfig, MBConvConfig
import torch
from torch import nn, Tensor
from functools import partial
from typing import Any

from const import height, width

def efficientnet_v2_xl(**kwargs: Any) -> EfficientNet:
    inverted_residual_setting = [
        FusedMBConvConfig(1, 3, 1, 32, 32, 4),
        FusedMBConvConfig(4, 3, 2, 32, 64, 8),
        FusedMBConvConfig(4, 3, 2, 64, 96, 8),
        MBConvConfig(4, 3, 2, 96, 192, 16),
        MBConvConfig(6, 3, 1, 192, 256, 24),
        MBConvConfig(6, 3, 2, 256, 512, 32),
        MBConvConfig(6, 3, 1, 512, 640, 8),
    ]
    last_channel = 1280
    dropout = 0.5

    model = EfficientNet(inverted_residual_setting, dropout, 
                         last_channel=last_channel,
                         norm_layer=partial(nn.BatchNorm2d, eps=1e-03), **kwargs)
    return model

def load_weight(model: EfficientNet, size='s') -> EfficientNet:
    import numpy as np
    import os

    weight_path = os.path.join('preweight',f'efficientnetv2-{size}-21k.npz')

    if not os.path.exists(weight_path):
        print('not found:', weight_path)
        return model

    with np.load(weight_path) as weights:
        print('loading weights')
        def apply_weights(func, base, tag=None):
            if isinstance(func, nn.Conv2d):
                state_dict = func.state_dict()
                for key in state_dict.keys():
                    if key == 'weight':
                        if tag:
                            target = base + tag
                            state_dict[key] = torch.from_numpy(weights[target]).permute(2,3,0,1)
                        else:
                            target = base + 'kernel'
                            state_dict[key] = torch.from_numpy(weights[target]).permute(3,2,0,1)
                func.load_state_dict(state_dict)
            elif isinstance(func, nn.BatchNorm2d):
                state_dict = func.state_dict()
                for key in state_dict.keys():
                    if key == 'weight':
                        target = base + 'gamma'
                        state_dict[key] = torch.from_numpy(weights[target])
                    elif key == 'bias':
                        target = base + 'beta'
                        state_dict[key] = torch.from_numpy(weights[target])
                    elif key == 'running_mean':
                        target = base + 'moving_mean'
                        state_dict[key] = torch.from_numpy(weights[target])
                    elif key == 'running_var':
                        target = base + 'moving_variance'
                        state_dict[key] = torch.from_numpy(weights[target])
                func.load_state_dict(state_dict)
                    
        idx = 0
        for i,section in enumerate(model.features):
            if i == 0:
                for func in section:
                    if isinstance(func, nn.Conv2d):
                        apply_weights(func, f'efficientnetv2-{size}/stem/conv2d/')
                    elif isinstance(func, nn.BatchNorm2d):
                        apply_weights(func, f'efficientnetv2-{size}/stem/tpu_batch_normalization/')
            elif i > 0 and not isinstance(section, Conv2dNormActivation):
                for sec in section:
                    if len(sec.block) == 1:
                        for func in sec.block[0]:
                            if isinstance(func, nn.Conv2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/conv2d/')
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/tpu_batch_normalization/')
                    elif len(sec.block) == 2:
                        for func in sec.block[0]:
                            if isinstance(func, nn.Conv2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/conv2d/')
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/tpu_batch_normalization/')
                        for func in sec.block[1]:
                            if isinstance(func, nn.Conv2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/conv2d_1/')
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/tpu_batch_normalization_1/')
                    elif len(sec.block) == 4:
                        for func in sec.block[0]:
                            if isinstance(func,nn.Conv2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/conv2d/')
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/tpu_batch_normalization/')
                        for func in sec.block[1]:
                            if isinstance(func, nn.Conv2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/depthwise_conv2d/', 'depthwise_kernel')
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/tpu_batch_normalization_1/')
                        apply_weights(sec.block[2].fc1, f'efficientnetv2-{size}/blocks_{idx}/se/conv2d/')
                        apply_weights(sec.block[2].fc2, f'efficientnetv2-{size}/blocks_{idx}/se/conv2d_1/')
                        for func in sec.block[3]:
                            if isinstance(func, nn.Conv2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/conv2d_1/')
                            elif isinstance(func, nn.BatchNorm2d):
                                apply_weights(func, f'efficientnetv2-{size}/blocks_{idx}/tpu_batch_normalization_2/')
                    idx += 1
            else:
                for func in section:
                    if isinstance(func, nn.Conv2d):
                        apply_weights(func, f'efficientnetv2-{size}/head/conv2d/')
                    elif isinstance(func, nn.BatchNorm2d):
                        apply_weights(func, f'efficientnetv2-{size}/head/tpu_batch_normalization/')
    return model

class BackboneModel(nn.Module):
    def __init__(self, model_size='s', pre_weights=True, **kwargs):
        super().__init__(**kwargs)
        if model_size == 's':
            model = efficientnet_v2_s(weights=None)
        elif model_size == 'm':
            model = efficientnet_v2_m(weights=None)
        elif model_size == 'l':
            model = efficientnet_v2_l(weights=None)
        elif model_size == 'xl':
            model = efficientnet_v2_xl()

        if pre_weights:
            model = load_weight(model, model_size)
        self.features = model.features

    def forward(self, x):
        results = []
        for ii,block in enumerate(self.features):
            x = block(x)
            if ii in [2,3,5]:
                results.append(x)
        results.append(x)
        return results

class Leafmap(nn.Module):
    def __init__(self, model_size='s', conv_dim = 256, out_dim=1, **kwargs) -> None:
        super().__init__(**kwargs)
        if model_size == 'xl':
            in_dims = [64,96,256,1280]
        elif model_size == 'l':
            in_dims = [64,96,224,1280]
        elif model_size == 'm':
            in_dims = [48,80,176,1280]
        elif model_size == 's':
            in_dims = [48,64,160,1280]
        global_dim = 1280
        self.global_bn = nn.BatchNorm2d(global_dim, eps=1e-03)
        self.in_bn = nn.ModuleList(
            [nn.BatchNorm2d(dim, eps=1e-03) for dim in in_dims]
        )
        upsamplers = []
        for i, in_dim in enumerate(reversed(in_dims)):
            if i == 0:
                layers = nn.Sequential(
                    nn.Conv2d(in_dim + global_dim, conv_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(conv_dim, eps=1e-03),
                    nn.SiLU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                )
            elif i > 0 and i < len(in_dims) - 1:
                layers = nn.Sequential(
                    nn.Conv2d(in_dim + conv_dim, conv_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(conv_dim, eps=1e-03),
                    nn.SiLU(),
                    nn.UpsamplingBilinear2d(scale_factor=2),
                )
            else:
                layers = nn.Sequential(
                    nn.Conv2d(in_dim + conv_dim, conv_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(conv_dim, eps=1e-03),
                    nn.SiLU(),
                )
            upsamplers.append(layers)
        self.upsamplers = nn.ModuleList(upsamplers)

        self.top_conv = nn.Sequential(
            nn.Conv2d(conv_dim, out_dim, 3, padding=1),
        )

    def forward(self, x0, x1, x2, x3, x4) -> Tensor:
        y = self.global_bn(x0)
        for x, bn, up in zip(reversed([x1,x2,x3,x4]), reversed(self.in_bn), self.upsamplers):
            x = bn(x)
            x = torch.cat([y,x], dim=1)
            y = up(x)
        return self.top_conv(y)

class AreaClassifyModel(nn.Module):
    def __init__(self, model_size='s', pre_weights=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.backbone = BackboneModel(model_size=model_size, pre_weights=pre_weights)
        self.classmap = Leafmap(model_size=model_size, conv_dim=64, out_dim=6)
        self.linemap = Leafmap(model_size=model_size, conv_dim=16, out_dim=1)
        self.decomap = Leafmap(model_size=model_size, conv_dim=16, out_dim=1)
        self.formulamap = Leafmap(model_size=model_size, conv_dim=16, out_dim=1)

        if model_size == 's':
            model = efficientnet_v2_s(weights=None)
        elif model_size == 'm':
            model = efficientnet_v2_m(weights=None)
        elif model_size == 'l':
            model = efficientnet_v2_l(weights=None)
        elif model_size == 'xl':
            model = efficientnet_v2_xl()

        if pre_weights:
            model = load_weight(model, model_size)
        self.globalmap = nn.Sequential(
            model.features,
            nn.UpsamplingBilinear2d(scale_factor=4),
        )

    def forward(self, x, gx):
        x = x * 2 - 1
        gx = gx * 2 - 1
        x = self.backbone(x)
        gx = self.globalmap(gx)
        h = height // 8
        w = width // 8
        gx = gx[:,:,h*3//8:-h*3//8,w*3//8:-w*3//8]
        return self.classmap(gx, *x), torch.concat([self.linemap(gx, *x), self.decomap(gx, *x), self.formulamap(gx, *x)], dim=1)

if __name__=="__main__":
    from torchinfo import summary

    # model = BackboneModel(model_size='s')
    # print(model)
    # with torch.no_grad():
    #     outputs = model(torch.zeros(1, 3, height, width))
    # print([t.shape for t in outputs])

    # exit()

    model = AreaClassifyModel(model_size='s', pre_weights=True)
    print(model)
    x = torch.zeros([1, 3, height, width])
    gx = torch.zeros([1, 3, height, width])
    y = model(x,gx)
    summary(model, input_data=(x, gx))
    with torch.no_grad():
        output = model(x,gx)
    print(output)
    print([o.shape for o in output])
    