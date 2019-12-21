import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import PIL
from PIL import Image
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import tensorboardX
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

import scipy.io
from models import ResDeepLab
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PIXEL_MEANS = [0.485, 0.456, 0.406]
PIXEL_VARS = [0.229, 0.224, 0.225]

writer = SummaryWriter()

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

def crop(size):
    crop = transforms.CenterCrop(size)
    return crop

to_tensor_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        PIXEL_MEANS,
        PIXEL_VARS
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

class DefaultDataset():
    def __init__(self, voc_dir, image_dir, segmentation_dir, names_dir, mode, batch_size):
        with open(os.path.join(voc_dir, names_dir, mode + '.txt'), 'r') as f:
            self.names = [line[:-1] for line in f]

        self.voc_dir = voc_dir
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.batch_size = batch_size

        
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        img = Image.open(os.path.join(self.voc_dir, self.image_dir, img_name + '.jpg'))
        img = to_tensor_normalize(img)

        seg = np.array(Image.open(os.path.join(self.voc_dir, self.segmentation_dir, img_name + '.png')))
        seg[seg == 255] = 0
        seg = Image.fromarray(seg).convert('P')
        seg = torch.from_numpy(np.array(seg.getchannel(0)))

        return img, seg


class PascalVOCDataset():
    def __init__(self, voc_dir, image_dir, segmentation_dir, names_dir, mode, batch_size=10, size=224):
        with open(os.path.join(voc_dir, names_dir, mode + '.txt'), 'r') as f:
            self.names = [line[:-1] for line in f]

        remains = len(self.names) % batch_size
        if remains != 0:
            self.names = self.names[:-remains]

        self.voc_dir = voc_dir
        self.image_dir = image_dir
        self.segmentation_dir = segmentation_dir
        self.mode = mode
        self.count = 0
        self.batch_size = batch_size
        self.scale = 1
        self.size = size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        img = Image.open(os.path.join(self.voc_dir, self.image_dir, img_name + '.jpg'))


        if self.count % self.batch_size == 0:
            self.scale = np.random.uniform(1.0, 1.2)
            self.scale = 1

        img = img_resize(self.size, self.scale)(img)

        p_flip = np.random.uniform(0, 1)
        flip = horizontal_flip(p_flip)
        img = flip(img)

        img = to_tensor_normalize(img)

        seg = np.array(Image.open(os.path.join(self.voc_dir, self.segmentation_dir, img_name + '.png')))
        seg[seg == 255] = 0
        seg = Image.fromarray(seg).convert('P')
        seg = seg_resize(self.size, self.scale)(seg)
        seg = flip(seg)
        seg = torch.from_numpy(np.array(seg.getchannel(0)))

        self.count = (self.count + 1) % len(self)
        return img, seg


class ContourDataset():
    def __init__(self, contour_dir, contour_img_dir, contour_names, batch_size=10, size=224):
        with open(contour_names, 'r') as f:
            self.names = [line[:-1] for line in f]

        remains = len(self.names) % batch_size
        if remains != 0:
            self.names = self.names[:-remains]

        self.contour_dir = contour_dir
        self.contour_img_dir = contour_img_dir
        self.count = 0
        self.batch_size = batch_size
        self.scale = 1
        self.size = size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img_name = self.names[idx]
        img = Image.open(os.path.join(self.contour_img_dir, img_name + '.jpg'))

        if self.count % self.batch_size == 0:
            self.scale = np.random.uniform(1.0, 1.2)
            self.scale = 1

        img = img_resize(self.size, self.scale)(img)

        p_flip = np.random.uniform(0, 1)
        flip = horizontal_flip(p_flip)
        img = flip(img)

        img = to_tensor_normalize(img)

        seg = scipy.io.loadmat(os.path.join(self.contour_dir, img_name + '.mat'))
        seg = seg['GTcls']['Segmentation'][0][0]
        seg[seg > 20] = 0
        seg = Image.fromarray(seg).convert('P')
        seg = seg_resize(self.size, self.scale)(seg)
        seg = flip(seg)
        seg = torch.from_numpy(np.array(seg.getchannel(0)))
        self.count = (self.count + 1) % len(self)
        return img, seg
        

class ExpandedVOCDataset():
    def __init__(self, voc_dir, image_dir, segmentation_dir, names_dir, mode, contour_dir, contour_img_dir,
                 contour_names, batch_size=10):
        self.WOK = PascalVOCDataset(voc_dir, image_dir, segmentation_dir, names_dir, mode, batch_size)
        self.contour = ContourDataset(contour_dir, contour_img_dir, contour_names, batch_size)
        self.first = np.arange(len(self.WOK))
        self.second = np.arange(len(self.contour))

        self.WOK_len = (len(self.WOK) // batch_size) * batch_size
        self.contour_len = (len(self.contour) // batch_size) * batch_size

    def __len__(self):
        return self.WOK_len + self.contour_len

    def __getitem__(self, idx):
        if idx < self.WOK_len:
            res = self.WOK[self.first[idx]]
        else:
            res = self.contour[self.second[idx - self.WOK_len]]

        if idx == len(self) - 1:
            np.random.shuffle(self.first)
            np.random.shuffle(self.second)

        return res


def pred_to_img(pred):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    r = Image.fromarray(pred.byte().cpu().numpy())
    r.putpalette(colors)
    r = r.convert('RGB')
    r = np.array(r)
    return torch.tensor(np.array(r))


def get_iou(pred, gt):
    if pred.shape != gt.shape:
        print('pred shape',pred.shape, 'gt shape', gt.shape)

    assert(pred.shape == gt.shape)
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = 20 
    count = np.zeros((max_label + 1,))
    for j in range(max_label + 1):
        x = np.where(pred == j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt == j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
 
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)
        
        if len(GT_idx_j) != 0:
            count[j] = float(len(n_jj)) / float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt))) 
    
    return Aiou


def calc_iou(out, seg):
    iou = 0
    for i in range(out.shape[0]):
        iou += get_iou(out.cpu().detach().numpy()[i], seg.cpu().detach().numpy()[i])

    return iou / out.shape[0]

def calc_iou1(out, seg, n_classes=21):
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

def calc_acc(out, seg):
    right = (out == seg).float()
    return right.mean(dim=(1, 2)).mean()


def evaluate(model, val_loader, iter_n):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    acc = 0
    iou = 0
    count = 0
    loss = 0
    for img, seg in val_loader:
        img = img.to(device)
        seg = seg.long().to(device)
        out = model(img)['out']
        loss += criterion(out, seg).item()        
        pred = torch.argmax(out, dim=1)
        acc += calc_acc(pred, seg)
        iou += calc_iou(pred, seg)

        if count == 0:
            writer.add_image('images/val/img', vutils.make_grid(img[0].detach(), normalize=True, scale_each=True), iter_n)
            writer.add_image('images/val/gt', pred_to_img(seg[0].detach()), iter_n, dataformats='HWC')
            writer.add_image('images/val/pred', pred_to_img(pred[0].detach()), iter_n, dataformats='HWC')

        count += 1

    return loss / count, acc / count, iou / count

def train_epoch_unary(train_loader, val_loader, model, opt, sch, iter_n=0):
    criterion = nn.CrossEntropyLoss()

    model.train()

    for data in train_loader:

        opt.zero_grad()
        img = data[0].to(device)
        seg = data[1].long().to(device)

        out = model(img)['out']
        loss = criterion(out, seg)
        loss.backward()
        opt.step()
        sch.step()

        train_pred = torch.argmax(out, dim=1)
        train_iou = calc_iou(train_pred, seg)
        train_acc = calc_acc(train_pred, seg)
        writer.add_scalar('train/loss', loss.item(), iter_n)
        writer.add_scalar('train/IoU', train_iou, iter_n)
        writer.add_scalar('train/acc', train_acc, iter_n)

        if iter_n % 100 == 0:
            writer.add_image('images/img', vutils.make_grid(img[0].detach(), normalize=True, scale_each=True), iter_n)
            writer.add_image('images/gt', pred_to_img(seg[0].detach()), iter_n, dataformats='HWC')
            writer.add_image('images/pred', pred_to_img(train_pred[0].detach()), iter_n, dataformats='HWC')
            model.train()
            print(iter_n)
        iter_n += 1
        
    return model, iter_n


def train(train_loader, val_loader, model, opt, sch, n_epochs, start_epoch=0, iter_n=0):
    since = time.time()
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('models/unary'):
        os.makedirs('models/unary')

    for i in range(n_epochs):
        model, iter_n = train_epoch_unary(train_loader, val_loader, model, opt, sch, iter_n)
        cur_time = time.time()
        print('Executed %d seconds' %(cur_time - since))

        torch.save(model.state_dict(), 'models/unary_' + str(start_epoch + i) + '.pth')
        val_loss, val_acc, val_iou = evaluate(model, val_loader, iter_n)
        writer.add_scalar('val/loss', val_loss, iter_n)
        writer.add_scalar('val/IoU', val_iou, iter_n)
        writer.add_scalar('val/acc', val_acc, iter_n)

    return model, iter_n

