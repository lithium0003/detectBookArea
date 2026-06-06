from torch.utils.data import IterableDataset, DataLoader
import pyvips
import numpy as np
from multiprocessing import shared_memory
import torch
import glob
import os
import math
import uuid
import time
import hashlib

from const import height, width

from .distortion import distortion

shm_map = {}

class AreaClassifyImageDataset(IterableDataset):
    def __init__(self, target_dir, loop_n=1, transform=False):
        super().__init__()
        self.target_dir = target_dir
        self.loop_n = loop_n
        self.labeled_images = sorted(glob.glob(os.path.join(target_dir, '*.labeled.png')))
        self.images = [f.replace('.labeled.png', '.npy') for f in self.labeled_images]
        self.line1_image = [f.replace('.labeled.png', '.rimline.png') for f in self.labeled_images]
        self.line2_image = [f.replace('.labeled.png', '.decorate.png') for f in self.labeled_images]
        self.line3_image = [f.replace('.labeled.png', '.formula.png') for f in self.labeled_images]
        invalid_idx = []
        print(f"Found {len(self.images)} images and {len(self.labeled_images)} labeled images.")
        for idx, img_path in enumerate(self.images):
            if not os.path.exists(img_path):
                print(f"Warning: Image file {img_path} does not exist. Skipping.")
                invalid_idx.append(idx)
        for idx in reversed(invalid_idx):
            del self.images[idx]
            del self.labeled_images[idx]
            del self.line1_image[idx]
            del self.line2_image[idx]
            del self.line3_image[idx]
        print(f"After validation, {len(self.images)} images and {len(self.labeled_images)} labeled images remain.")
        self.transform = transform

        self.ids_shm = None
        self.image_shm = []
        self.labeled_shm = []
        self.line1_shm = []
        self.line2_shm = []
        self.line3_shm = []
        self.shm_info = []

        self.ids = None
        self.image_buf = []
        self.label_buf = []
        self.line1_buf = []
        self.line2_buf = []
        self.line3_buf = []
        self.shm_buf = []
        self.prepared = False

        self.creater = os.getpid()
        ids = np.arange(self.len).astype(np.int64)
        sz = np.prod(ids.shape + (np.dtype("int64").itemsize,)).item()
        shm_name = hashlib.md5(uuid.uuid4().hex.encode()).hexdigest()[:30]
        shm = shared_memory.SharedMemory(name=shm_name, create=True, size=sz)
        ids2 = np.ndarray(shape=ids.shape, dtype=np.int64, buffer=shm.buf)
        ids2[:] = ids[:]
        self.ids_shm = shm
        self.shm_info.append((-1, shm_name, ids.shape))

        print('memory load')
        for i, image_name in enumerate(self.images):
            print(i, image_name)
            image = np.load(image_name)
            sz = np.prod(image.shape + (np.dtype("uint8").itemsize,)).item()
            shm_name = hashlib.md5(image_name.encode()).hexdigest()[:30]
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=sz)
                image2 = np.ndarray(shape=image.shape, dtype=np.uint8, buffer=shm.buf)
                image2[:] = image[:]
                self.image_shm.append(shm)
            except FileExistsError:
                pass
            self.shm_info.append((0, shm_name, image.shape))
        for i, (image_name1, image_name2, image_name3, image_name4) in enumerate(zip(self.labeled_images, self.line1_image, self.line2_image, self.line3_image)):
            print(i, image_name1)
            image = pyvips.Image.new_from_file(image_name1).numpy()
            sz = np.prod(image.shape + (np.dtype("uint8").itemsize,)).item()
            shm_name = hashlib.md5(image_name1.encode()).hexdigest()[:30]
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=sz)
                image2 = np.ndarray(shape=image.shape, dtype=np.uint8, buffer=shm.buf)
                image2[:] = image[:]
                self.labeled_shm.append(shm)
            except FileExistsError:
                pass
            self.shm_info.append((1, shm_name, image.shape))
    
            line1_image = np.zeros_like(image)
            if os.path.exists(image_name2):
                line1_image = pyvips.Image.new_from_file(image_name2).numpy()
            sz = np.prod(line1_image.shape + (np.dtype("uint8").itemsize,)).item()
            shm_name = hashlib.md5(image_name2.encode()).hexdigest()[:30]
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=sz)
                image2 = np.ndarray(shape=line1_image.shape, dtype=np.uint8, buffer=shm.buf)
                image2[:] = line1_image[:]
                self.line1_shm.append(shm)
            except FileExistsError:
                pass
            self.shm_info.append((2, shm_name, image.shape))

            line2_image = np.zeros_like(image)
            if os.path.exists(image_name3):
                line2_image = pyvips.Image.new_from_file(image_name3).numpy()
            sz = np.prod(line2_image.shape + (np.dtype("uint8").itemsize,)).item()
            shm_name = hashlib.md5(image_name3.encode()).hexdigest()[:30]
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=sz)
                image2 = np.ndarray(shape=line2_image.shape, dtype=np.uint8, buffer=shm.buf)
                image2[:] = line2_image[:]
                self.line2_shm.append(shm)
            except FileExistsError:
                pass
            self.shm_info.append((3, shm_name, image.shape))

            line3_image = np.zeros_like(image)
            if os.path.exists(image_name4):
                line3_image = pyvips.Image.new_from_file(image_name4).numpy()
            sz = np.prod(line3_image.shape + (np.dtype("uint8").itemsize,)).item()
            shm_name = hashlib.md5(image_name4.encode()).hexdigest()[:30]
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=True, size=sz)
                image2 = np.ndarray(shape=line3_image.shape, dtype=np.uint8, buffer=shm.buf)
                image2[:] = line3_image[:]
                self.line3_shm.append(shm)
            except FileExistsError:
                pass
            self.shm_info.append((4, shm_name, image.shape))

        print('memory load end')

    def __del__(self):
        print("del")
        if len(self.shm_buf) > 0:
            print("close shm")
            for shm in self.shm_buf:
                shm.close()
            self.shm_buf = []
            self.ids = None
            self.image_buf = []
            self.label_buf = []
            self.line1_buf = []
            self.line2_buf = []
            self.line3_buf = []
            self.prepared = False
        if os.getpid() == self.creater:
            print("unlink shm")
            if self.ids_shm is not None:
                self.ids_shm.close()
                self.ids_shm.unlink()
            for shm in self.image_shm:
                shm.close()
                shm.unlink()
            for shm in self.labeled_shm:
                shm.close()
                shm.unlink()
            for shm in self.line1_shm:
                shm.close()
                shm.unlink()
            for shm in self.line2_shm:
                shm.close()
                shm.unlink()
            for shm in self.line3_shm:
                shm.close()
                shm.unlink()

    def prepare(self):
        if self.prepared:
            return
        print('shm load start')
        self.prepared = True
        for i, image_name, shape in self.shm_info:
            shm = shared_memory.SharedMemory(name=image_name)
            if i == -1:
                self.ids = np.ndarray(shape=shape, dtype=np.int64, buffer=shm.buf)
            else:
                image = np.ndarray(shape=shape, dtype=np.uint8, buffer=shm.buf)
                if i == 0:
                    self.image_buf.append(image)
                elif i == 1:
                    self.label_buf.append(image)
                elif i == 2:
                    self.line1_buf.append(image)
                elif i == 3:
                    self.line2_buf.append(image)
                elif i == 4:
                    self.line3_buf.append(image)
            self.shm_buf.append(shm)
        print('shm load end')

    @property
    def weight(self):
        class_counts = np.zeros(6, dtype=np.int64)
        for label_image in self.labeled_images:
            label_image = np.asarray(pyvips.Image.new_from_file(label_image))
            class_counts = np.add(class_counts, np.bincount(label_image.flatten(), minlength=6))
        class_weights = 1 / (class_counts + 1e-6)
        return class_weights / class_weights[0]

    def generate(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            for idx in self.ids:
                yield self.get(idx)
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(self.len / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.len)
            for idx in self.ids[iter_start:iter_end]:
                yield self.get(idx)

    def __iter__(self):
        self.prepare()
        if self.transform:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None or worker_info.num_workers == 1 or worker_info.id == 0:
                np.random.shuffle(self.ids)
            else:
                time.sleep(1)
        return iter(self.generate())

    @property
    def len(self):
        return len(self.labeled_images) * self.loop_n

    def get(self, idx):
        image = self.image_buf[idx % len(self.image_buf)]
        label_image = self.label_buf[idx % len(self.label_buf)]
        line1_image = self.line1_buf[idx % len(self.line1_buf)]
        line2_image = self.line2_buf[idx % len(self.line2_buf)]
        line3_image = self.line3_buf[idx % len(self.line3_buf)]

        (h, w, _) = image.shape
        min_ratio = max(1.0 , min(width / w, height / h) * 1.5)
        bg_color = np.random.randint(0, 255, (3,)).astype(np.uint8)
        rotation_angle = np.random.normal() * 15
        aspect_ratio = np.random.uniform(0.5, 2.0)
        width_ratio = np.exp(np.random.uniform(np.log(min_ratio), np.log(2.0)))
        height_ratio = aspect_ratio * width_ratio

        rx = 1/width_ratio
        ry = 1/height_ratio
        x = w // 2
        y = h // 2
        delta_x = int(w * 0.6 * width_ratio)
        delta_y = int(h * 0.6 * height_ratio)
        x_inv = np.random.randint(-delta_x, delta_x) + x
        y_inv = np.random.randint(-delta_y, delta_y) + y
        angle = rotation_angle / 180 * np.pi
        move_matrix = np.array([
            [1, 0, -x_inv],
            [0, 1, -y_inv],
            [0, 0, 1],
        ])
        rotate_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [            0,              0, 1],
        ])
        moveback_matrix = np.array([
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1],
        ])
        scale_matrix = np.array([
            [rx,  0, 0],
            [ 0, ry, 0],
            [ 0,  0, 1],
        ])
        M = np.dot(moveback_matrix, rotate_matrix)
        M = np.dot(M, scale_matrix)
        M = np.dot(M, move_matrix)
        M = M.astype(np.float32)

        if self.transform:
            brightness_factor = np.random.uniform(-0.2, 0.2)
            contrast_factor = np.random.uniform(0.8, 1.25)
            saturation_factor = np.random.uniform(0.8, 1.25)
            hue_factor = np.random.uniform(-0.5, 0.5)
        else:
            brightness_factor = 0.0
            contrast_factor = 1.0
            saturation_factor = 1.0
            hue_factor = 0.0

        im_small, im_large, im2_out, im3_out, im4_out, im5_out = distortion(image, label_image, line1_image, line2_image, line3_image, M, bg_color, width, height, brightness_factor, contrast_factor, saturation_factor, hue_factor)

        small_image = im_small.astype(np.float32).transpose(2, 0, 1) / 255.
        large_image = im_large.astype(np.float32).transpose(2, 0, 1) / 255.
        label_image = im2_out.astype(np.long)
        line1_image = np.stack([im3_out.astype(np.float32), im4_out.astype(np.float32), im5_out.astype(np.float32)], axis=0) / 255.

        return small_image, large_image, label_image, line1_image

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time

    start = time.time()
    training_dataset = AreaClassifyImageDataset('data', loop_n=128, transform=True)
    training_loader = DataLoader(training_dataset, batch_size=32, num_workers=os.cpu_count(), persistent_workers=True, in_order=False)
    count = 0
    for small_image, large_image, label_image, line1_image in training_loader:
        print(count)
        print(small_image.shape, large_image.shape, label_image.shape, line1_image.shape)
        count += 1
        print(time.time() - start)
        start = time.time()
        # continue
        plt.figure()
        for i in range(4):
            plt.subplot(6, 4, i+1)
            plt.imshow(small_image[i].numpy().transpose(1, 2, 0))
            plt.subplot(6, 4, i+5)
            plt.imshow(label_image[i].numpy())
            plt.subplot(6, 4, i+9)
            plt.imshow(large_image[i].numpy().transpose(1, 2, 0))
            plt.subplot(6, 4, i+13)
            plt.imshow(line1_image[i][0].numpy())
            plt.subplot(6, 4, i+17)
            plt.imshow(line1_image[i][1].numpy())
            plt.subplot(6, 4, i+21)
            plt.imshow(line1_image[i][2].numpy())
        plt.show()
        # break
