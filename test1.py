#!/usr/bin/env python3

import torch
from models.area_classify import AreaClassifyModel
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from const import height, width
width4 = width * 4
height4 = height * 4

def test_area_classify(target_file=None):
    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    # device = torch.device('cpu')
    print('using device:', device, flush=True)

    if os.path.exists('ave_model.pt'):
        data = torch.load('ave_model.pt', map_location="cpu", weights_only=True)
        model = AreaClassifyModel(pre_weights=False)
        model.load_state_dict(data['model_state_dict'])
        print('loaded ave_model')
    elif os.path.exists('result1/model.pt'):
        data = torch.load('result1/model.pt', map_location="cpu", weights_only=True)
        model = AreaClassifyModel(pre_weights=False)
        model.load_state_dict(data['model_state_dict'])
        print('loaded epoch:', data['epoch'])
    else:
        model = AreaClassifyModel(pre_weights=True)
    model.to(device)
    model.eval()

    if target_file is None:
        return

    image0 = Image.open(target_file)
    image0 = np.asarray(image0)
    if len(image0.shape) < 3:
        image0 = np.stack([image0,image0,image0],axis=-1)
    else:
        image0 = image0[:,:,:3]
    if image0.dtype.name == 'uint8':
        vscale = 255.0
    else:
        vscale = 65535.0
    print(image0.shape)
    # image0 = ndimage.zoom(image0, (1.5, 1.5, 1), order=1)
    # print(image0.shape)
    h0, w0, _ = image0.shape
    image1 = np.pad(image0, (((height4-height)//2, max(0, height4 - image0.shape[0])), ((width4-width)//2, max(0, width4 - image0.shape[1])), (0, 0)))
    h, w, _ = image1.shape
    pred0 = np.zeros(((h + height4 - 1) // 4, (w + width4 - 1) // 4), dtype=np.uint8)
    pred1 = np.zeros(((h + height4 - 1) // 4, (w + width4 - 1) // 4))
    pred2 = np.zeros(((h + height4 - 1) // 4, (w + width4 - 1) // 4))
    pred3 = np.zeros(((h + height4 - 1) // 4, (w + width4 - 1) // 4))
    for x in range(0, w, width):
        for y in range(0, h, height):
            print(f"Processing area x:{x}-{x+width}, y:{y}-{y+height}", flush=True)
            large_image = image1[y:y+height4, x:x+width4, :]
            large_image = np.pad(large_image, ((0, max(0, height4 - large_image.shape[0])), (0, max(0, width4 - large_image.shape[1])), (0, 0)))
            small_image = large_image[(height4-height)//2:-(height4-height)//2, (width4-width)//2:-(width4-width)//2, :]
            small_image = small_image.transpose(2,0,1).astype(np.float32) / vscale
            large_image = ndimage.zoom(large_image, (0.25, 0.25, 1), order=1).transpose(2,0,1).astype(np.float32) / vscale
            # plt.subplot(2,2,1)
            # plt.imshow(small_image.transpose(1,2,0))
            # plt.subplot(2,2,2)
            # plt.imshow(large_image.transpose(1,2,0))
            small_image = torch.tensor(small_image).unsqueeze(0).to(device)
            large_image = torch.tensor(large_image).unsqueeze(0).to(device)
            with torch.no_grad():
                output, line_output = model(small_image, large_image)
                pred, idx = output.squeeze(0).softmax(axis=0).max(axis=0)
                output_image = torch.where(pred > 0.66, idx, 0).cpu().numpy()
            line_image1, line_image2, line_image3 = torch.sigmoid(line_output).squeeze(0).chunk(3)
            line_image1 = line_image1.cpu().numpy()
            line_image2 = line_image2.cpu().numpy()
            line_image3 = line_image3.cpu().numpy()
            # plt.subplot(2,2,3)
            # plt.imshow(output_image)
            # plt.subplot(2,2,4)
            # plt.imshow(line_image1.transpose(1,2,0))
            # plt.show()
            pred0[y//4:(y+height)//4, x//4:(x+width)//4] = output_image
            pred1[y//4:(y+height)//4, x//4:(x+width)//4] = line_image1
            pred2[y//4:(y+height)//4, x//4:(x+width)//4] = line_image2
            pred3[y//4:(y+height)//4, x//4:(x+width)//4] = line_image3
    pred_image = pred0.astype('uint8')[:h0//4+1,:w0//4+1]
    line1_image = pred1[:h0//4+1,:w0//4+1]
    line2_image = pred2[:h0//4+1,:w0//4+1]
    line3_image = pred3[:h0//4+1,:w0//4+1]
    pred_image = ndimage.zoom(pred_image, 4, order=0)
    line1_image = ndimage.zoom(line1_image, 4, order=1)
    line2_image = ndimage.zoom(line2_image, 4, order=1)
    line3_image = ndimage.zoom(line3_image, 4, order=1)
    fig, ax = plt.subplots(2, 3, sharex="all", sharey="all")
    ax[0,0].imshow(image0.astype(np.float32) / vscale)
    ax[0,1].imshow(pred_image)
    ax[0,2].remove()
    ax[1,0].imshow(line1_image)
    ax[1,1].imshow(line2_image)
    ax[1,2].imshow(line3_image)
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_area_classify(sys.argv[1])
    else:
        test_area_classify()
