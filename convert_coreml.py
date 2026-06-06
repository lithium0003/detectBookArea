#!/usr/bin/env python3
import coremltools as ct
import torch
import os
from datetime import datetime

from models.area_classify import AreaClassifyModel
from const import height, width

def convert():
    if os.path.exists('result1/model.pt'):
        data = torch.load('result1/model.pt', map_location="cpu", weights_only=True)
        model = AreaClassifyModel(pre_weights=False)
        model.load_state_dict(data['model_state_dict'])
        print('loaded epoch:', data['epoch'])
    else:
        model = AreaClassifyModel(pre_weights=True)
    model.eval()

    class MLProxyModel(torch.nn.Module):
        def __init__(self, base, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.base = base

        def forward(self, x, gx):
            classmap, linemap = self.base(x, gx)
            classmap = torch.softmax(classmap, 1)
            maxval, classidx  = torch.max(classmap, dim=1)
            classmap = torch.where(maxval > 0.66, classidx, 0)
            linemap1, linemap2, linemap3 = torch.sigmoid(linemap).chunk(3, dim=1)
            return classmap, linemap1.squeeze(1), linemap2.squeeze(1), linemap3.squeeze(1)

    textarea_model = MLProxyModel(model)
    textarea_model.eval()

    small_input = torch.rand(1, 3, height, width)
    large_input = torch.rand(1, 3, height, width)
    traced_model = torch.jit.trace(textarea_model, (small_input, large_input))

    mlmodel_textarea = ct.convert(traced_model,
            inputs=[
                ct.ImageType(name='small', shape=(1, 3, height, width), scale=1/255),
                ct.ImageType(name='large', shape=(1, 3, height, width), scale=1/255),
            ],
            outputs=[
                ct.TensorType(name='classmap'),
                ct.TensorType(name='line'),
                ct.TensorType(name='deco'),
                ct.TensorType(name='formula'),
            ],
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS26)
    mlmodel_textarea.version = datetime.now().strftime("%Y%m%d%H%M%S")
    mlmodel_textarea.save("TextArea.mlpackage")

if __name__ == "__main__":
    convert()
