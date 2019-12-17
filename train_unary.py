import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import cv2
import PIL
from PIL import Image
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable

from models import ResDeepLab
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIXEL_MEANS = [104.008, 116.669, 122.675]


voc_dir = 'VOC2012'
image_dir = 'JPEGImages'
segmentation_dir = 'SegmentationClass'
names_dir = 'ImageSets/Segmentation'


def img_resize(size, scale):
    new_size = int(size * scale)
    return transforms.Resize((new_size, new_size), PIL.Image.BILINEAR)

def seg_resize(size, scale):
    new_size = int(size * scale)
    return transforms.Resize((new_size, new_size), PIL.Image.NEAREST)

def horizontal_flip(p):
    if p > 0.5:
        return transforms.RandomHorizontalFlip(1.0)

    return transforms.RandomHorizontalFlip(0.0)


to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        PIXEL_MEANS,
        [1.0, 1.0, 1.0]
    )
])

to_tensor = transforms.ToTensor()

def down_size(in_size):
    out_size = int(in_size)
    out_size = (out_size + 1) // 2
    out_size = int(np.ceil((out_size + 1) / 2.0))
    out_size = (out_size + 1) // 2
    return out_size


def down_seg(seg, size):
    res = F.interpolate(seg[None, None, :, :],
        size=(size, size),
    )

    return res[0]


class PascalVOCDataset():
    def __init__(self, voc_dir, image_dir, segmentation_dir, names_dir, mode, batch_size=10):
        with open(os.path.join(voc_dir, names_dir, mode + '.txt'), 'r') as f:
            self.names = [line[:-1] for line in f]

        self.voc_dir = voc_dir
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.mode = mode
        self.count = 0
        self.batch_size = batch_size
        self.scale = 1

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        img = Image.open(os.path.join(self.voc_dir, self.image_dir, img_name + '.jpg'))


        if self.count % self.batch_size == 0:
            self.scale = np.random.uniform(0.5, 1.3)

        img = img_resize(321, self.scale)(img)

        p_flip = np.random.uniform(0, 1)
        flip = horizontal_flip(p_flip)
        img = flip(img)

        img = to_tensor_normalize(img)

        seg = np.array(Image.open(os.path.join(self.voc_dir, self.segmentation_dir, img_name + '.png')))
        seg[seg == 255] = 0
        seg = Image.fromarray(seg).convert('P')
        seg = seg_resize(321, self.scale)(seg)
        seg = flip(seg)
        seg = seg_resize(down_size(int(321 * self.scale)), 1)(seg)
        seg = torch.from_numpy(np.array(seg.getchannel(0)))

        self.count = (self.count + 1) % len(self)
        return img, seg

batch_size = 2
train_voc_dataset = PascalVOCDataset(voc_dir, image_dir, segmentation_dir, names_dir, 'train', batch_size=batch_size)
train_voc_loader = torch.utils.data.DataLoader(train_voc_dataset, batch_size=batch_size, shuffle=True)

val_voc_dataset = PascalVOCDataset(voc_dir, image_dir, segmentation_dir, names_dir, 'val', batch_size=batch_size)
val_voc_loader = torch.utils.data.DataLoader(val_voc_dataset, batch_size=batch_size, shuffle=True)


def calc_iou(out, seg, n_classes=21):
    if out.shape != seg.shape:
        print('out shape', out.shape)
        print('seg shape', seg.shape)

    assert(out.shape == seg.shape)
    ious = torch.zeros(out.shape[0], n_classes - 1, dtype=torch.float)

    for label in range(1, n_classes):
        pred_idxs = out == label
        target_idxs = seg == label
        intersection = pred_idxs * target_idxs
        intersection = intersection.long().sum(dim=(1, 2))
        union = pred_idxs.long().sum(dim=(1, 2)) + target_idxs.long().sum(dim=(1, 2)) - intersection
        cur_ious = -torch.ones_like(union, dtype=torch.float)
        cur_ious[union > 0] = intersection[union > 0].float() / union[union > 0].float()
        ious[:, label - 1] = cur_ious

    res = torch.zeros(out.shape[0])
    for i in range(out.shape[0]):
        valid_ious = ious[i]
        valid_ious = valid_ious[valid_ious > -1]

        if len(valid_ious) == 0:
            res[i] = 1
        else:
            res[i] = valid_ious[valid_ious > -1].mean()

    return res.mean()


def train_epoch_unary(train_loader, model, opt):
    criterion = nn.CrossEntropyLoss()

    model.train()
    losses = []
    i = 0
    for data in train_loader:
        i += 1
        if i % 10 == 0:
            break
        opt.zero_grad()
        img = data[0].to(device)
        seg = data[1].long().to(device)

        out = model(img)
        loss = criterion(out, seg)
        loss.backward()
        opt.step()

        pred = torch.argmax(out, dim=1)
        print(loss.item())
        print(calc_iou(pred, seg, 21))

        losses.append(loss.item())

    return model, losses


def train_unary(train_loader, model, opt, n_epochs):
    since = time.time()
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('models/unary'):
        os.makedirs('models/unary')

    for i in range(n_epochs):
        model, losses = train_epoch_unary(train_loader, model, opt)
        cur_time = time.time()
        print('Executed %d seconds' %(cur_time - since))

        torch.save(model.state_dict(), 'models/unary_' + str(i) + '.pth')

    return model


model = ResDeepLab().to(device)
opt = torch.optim.Adam(model.parameters(), lr=3e-4)

model = train_unary(train_voc_loader, model, opt, 1)

