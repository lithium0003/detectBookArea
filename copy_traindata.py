import glob
import os
import shutil
import pyvips
import numpy as np

source_dirs = ['data1','data2','data3','data4','data5','data6']
target_dir = 'data'

os.makedirs(target_dir, exist_ok=True)
i = 0
for source_dir in source_dirs:
    files = glob.glob(os.path.join(source_dir, '*.labeled.png'))
    for file in sorted(files):
        print(file)
        basename = file.replace('.labeled.png','')
        ext = os.path.splitext(basename)[1]
        target = os.path.join(target_dir, f'image{i:08d}{ext}')
        image = pyvips.Image.new_from_file(basename).numpy()
        if len(image.shape) == 2:
            image = np.tile(image[:,:,None],(1,1,3))
        else:
            image = image[:,:,:3]
        if image.dtype.name != 'uint8':
            image = (image / 255).astype(np.unit8)
        np.save(target, image)
        labeled = file
        shutil.copyfile(labeled, target+'.labeled.png')
        decorate = basename+'.decorate.png'
        if os.path.exists(decorate):
            shutil.copyfile(decorate, target+'.decorate.png')
        rimline = basename+'.rimline.png'
        if os.path.exists(rimline):
            shutil.copyfile(rimline, target+'.rimline.png')
        formula = basename+'.formula.png'
        if os.path.exists(formula):
            shutil.copyfile(formula, target+'.formula.png')
        i += 1