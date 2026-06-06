#!/usr/bin/env python3

import coremltools as ct

import numpy as np
import sys
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

mlmodel_textarea = ct.models.MLModel('TextArea.mlpackage')

def test_area_classify(target_file=None):
    if target_file is None:
        return

    image0 = Image.open(target_file)
    image0 = np.asarray(image0)
    if len(image0.shape) < 3:
        image0 = np.stack([image0,image0,image0],axis=-1)
    else:
        image0 = image0[:,:,:3]
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
            if small_image.dtype.name != 'uint8':
                small_image = (small_image / 255).astype(np.unit8)
            large_image = ndimage.zoom(large_image, (0.25, 0.25, 1), order=1)
            if small_image.dtype.name != 'uint8':
                large_image = (large_image / 255).astype(np.unit8)
            small_image = Image.fromarray(small_image)
            large_image = Image.fromarray(large_image)
            # plt.subplot(2,2,1)
            # plt.imshow(small_image)
            # plt.subplot(2,2,2)
            # plt.imshow(large_image)
            output = mlmodel_textarea.predict({'small': small_image, 'large': large_image})
            output_image = output['classmap'][0]
            line_image1 = output['line'][0]
            line_image2 = output['deco'][0]
            line_image3 = output['formula'][0]
            # plt.subplot(2,2,3)
            # plt.imshow(output_image)
            # plt.subplot(2,2,4)
            # plt.imshow(line_image1)
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
    ax[0,0].imshow(image0)
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
