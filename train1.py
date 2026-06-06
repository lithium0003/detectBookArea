#!/usr/bin/env python3

## ulimit -Sn 1048576
import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
print('soft limit:', soft)
print('hard limit:', hard)
resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

import torch
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
import os
import sys
import datetime

from models.adamw_schedulefree import AdamWScheduleFree
from models.area_classify import AreaClassifyModel
from dataset.dataset_classify import AreaClassifyImageDataset

lr = 1e-4
wd = 1e-4
EPOCHS = 500
batch=128
logstep = 1

class RunningLoss(torch.nn.modules.Module):
    def __init__(self, *args, **kwargs) -> None:
        self.device = kwargs.pop('device', 'cpu')
        self.step = 0
        self.count = 0
        self.writer = SummaryWriter(log_dir="result1/logs")
        self.runningcount = kwargs.pop('runningcount', 1000)
        self.losses = kwargs.pop('losses', [])
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        if self.count > 0:
            self.write()

        self.count = 0
        self.running_loss = {key: torch.tensor(0., dtype=torch.float, device=self.device) for key in self.losses}
        self.correct = torch.tensor(0, device=self.device)
        self.total = torch.tensor(0, device=self.device)

    def write(self, ret=None):
        if ret is None:
            ret = {}
            for key in self.losses:
                ret[key] = self.running_loss[key] / self.count if self.count > 0 else 0.
            ret['accuracy'] = self.correct.float() / self.total.float() if self.total > 0 else torch.tensor(0., dtype=torch.float, device=self.device)

        for key in ret:
            name = 'train/'+key if self.training else 'val/'+key
            self.writer.add_scalar(name, ret[key], self.step)

        return ret

    def forward(self, losses):
        if self.training:
            self.step += 1
        self.count += 1
        for key in self.losses:
            self.running_loss[key] += losses[key]
        self.correct += losses['correct']
        self.total += losses['total']

        ret = {}
        for key in self.losses:
            ret[key] = self.running_loss[key] / self.count if self.count > 0 else 0.
        ret['accuracy'] = self.correct.float() / self.total.float() if self.total > 0 else torch.tensor(0., dtype=torch.float, device=self.device)
        if 'lr' in losses:
            ret['lr'] = losses['lr']

        if self.training and self.count % self.runningcount == 0:
            self.write(ret)
            self.reset()

        return ret

def train():
    training_dataset = AreaClassifyImageDataset('data', loop_n=32, transform=True)
    training_loader = DataLoader(training_dataset, batch_size=batch, num_workers=min(batch, 16, os.cpu_count()), persistent_workers=True, in_order=False, pin_memory=True, drop_last=True)
    class_weights = torch.tensor(training_dataset.weight, dtype=torch.float32)
    print("Class weights:", class_weights)

    validate_dataset = AreaClassifyImageDataset('data')
    validate_loader = DataLoader(validate_dataset, batch_size=batch, num_workers=min(batch, 4, os.cpu_count()), persistent_workers=True, in_order=False)

    device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print('using device:', device, flush=True)

    if os.path.exists('result1/model.pt'):
        data = torch.load('result1/model.pt', map_location="cpu", weights_only=True)
        model = AreaClassifyModel(pre_weights=False)
        model.load_state_dict(data['model_state_dict'])
    else:
        model = AreaClassifyModel(pre_weights=True)
    
    model.to(device) 
    optimizer = AdamWScheduleFree(model.parameters(), lr=lr, weight_decay=wd)
    class_weights = class_weights.to(device)

    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    running_loss = RunningLoss(device=device, runningcount=10, losses=[
        'loss',
        'class_loss',
        'line_loss',
    ])

    def loss_function(label, line1_image, heatmap, linemap):
        class_loss = torch.nn.functional.cross_entropy(heatmap, label, weight=class_weights)
        correct = (heatmap.argmax(dim=1) == label).sum()
        total = label.numel()
        line_loss = torch.nn.functional.binary_cross_entropy_with_logits(linemap, line1_image)
        loss = class_loss + line_loss
        return loss, class_loss, line_loss, correct, total

    def train_step(small_image, large_image, label_image, line1_image):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            heatmap, linemap = model(small_image, large_image)
            return loss_function(label_image, line1_image, heatmap, linemap)

    def test_step(small_image, large_image, label_image, line1_image):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            heatmap, linemap = model(small_image, large_image)
            return loss_function(label_image, line1_image, heatmap, linemap )

    if device.type == 'cuda':
        train_step = torch.compile(train_step, mode="reduce-overhead")
        test_step = torch.compile(test_step, mode="reduce-overhead")

    for epoch in range(EPOCHS):
        print(datetime.datetime.now(), 'epoch', epoch, flush=True)
        print(datetime.datetime.now(), 'lr', optimizer.param_groups[0]['lr'], flush=True)

        model.train()
        running_loss.train()
        optimizer.train()

        for i, data in enumerate(training_loader):
            small_image, large_image, label_image, line1_image = data
            small_image = small_image.to(device)
            large_image = large_image.to(device)
            label_image = label_image.to(device)
            line1_image = line1_image.to(device)
            optimizer.zero_grad()
            loss, class_loss, line_loss, correct, total = train_step(small_image, large_image, label_image, line1_image)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losslog = running_loss({'loss': loss, 'class_loss': class_loss, 'line_loss': line_loss, 'correct': correct, 'total': total, 'lr': optimizer.param_groups[0]['lr']})

            if (i + 1) % logstep == 0 or i == 0:
                loss_value = losslog['loss'].item()
                class_loss_value = losslog['class_loss'].item()
                line_loss_value = losslog['line_loss'].item()
                acc_value = losslog['accuracy'].item()
                print(epoch, i+1, datetime.datetime.now(), 'loss', loss_value, 'class_loss', class_loss_value, 'line_loss', line_loss_value, 'acc', acc_value, flush=True)

        running_loss.reset()
        model.eval()
        running_loss.eval()
        optimizer.eval()

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, 'result1/model.pt')

        with torch.no_grad():
            for i, data in enumerate(validate_loader):
                small_image, large_image, label_image, line1_image = data
                small_image = small_image.to(device)
                large_image = large_image.to(device)
                label_image = label_image.to(device)
                line1_image = line1_image.to(device)
                loss, class_loss, line_loss, correct, total = test_step(small_image, large_image, label_image, line1_image)
                losslog = running_loss({'loss': loss, 'class_loss': class_loss, 'line_loss': line_loss, 'correct': correct, 'total': total})

                if (i + 1) % logstep == 0 or i == 0:
                    loss_value = losslog['loss'].item()
                    class_loss_value = losslog['class_loss'].item()
                    line_loss_value = losslog['line_loss'].item()
                    acc_value = losslog['accuracy'].item()
                    print(epoch, i+1, datetime.datetime.now(), 'val_loss', loss_value, 'val_class_loss', class_loss_value, 'val_line_loss', line_loss_value, 'val_acc', acc_value, flush=True)
    
        running_loss.reset()

if __name__=='__main__':
    if len(sys.argv) > 1:
        argv = sys.argv[1:]
        for arg in argv:
            if arg.startswith('--epoch'):
                EPOCHS = int(arg.split('=')[1])
            elif arg.startswith('--workers'):
                workers = int(arg.split('=')[1])
            elif arg.startswith('--lr'):
                lr = float(arg.split('=')[1])
            elif arg.startswith('--logstep'):
                logstep = int(arg.split('=')[1])
            else:
                batch = int(arg)

    train()