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
from deeplabv3 import deeplabv3_resnet101
from layers import PottsTypeConjugateGradients, potts_type_crf, ConjugateGradients, dense_gaussian_crf

import scipy.io
import time

import train_unary
from train_unary import calc_acc, calc_iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = train_unary.writer

def load_pairwise(weights_file, embedding_size, n_classes=21, last_layer=4):
    pairwise = deeplabv3_resnet101()
    pairwise.load_state_dict(torch.load(weights_file))
    if last_layer == 4:
        pairwise.classifier = nn.Conv2d(2048, embedding_size * n_classes, kernel_size=(1, 1), bias=False)
    elif last_layer == 3:
        pairwise.backbone.layer4 = nn.Identity()
        pairwise.classifier = nn.Conv2d(1024, embedding_size * n_classes, kernel_size=(1, 1), bias=False)
    elif last_layer == 2:
        pairwise.backbone.layer3 = nn.Identity()
        pairwise.backbone.layer4 = nn.Identity()
        pairwise.classifier = nn.Conv2d(512, embedding_size * n_classes, kernel_size=(1, 1), bias=False)
    else:
        raise ValueError('Expected last_layer to take value in {2, 3, 4}')

    pairwise.to(device)
    return pairwise


def train_epoch_final(train_loader, unary, pairwise, opt, sch, iter_n=0, n_classes=21):
    criterion = nn.CrossEntropyLoss()
    solver = ConjugateGradients(shift=0.2)
    crf = dense_gaussian_crf(solver)

    unary.train()
    pairwise.train()
    
    for data in train_loader:
        opt.zero_grad()
        img = data[0].to(device)
        seg = data[1].long().to(device)

        unary_out = unary(img)['out']
        B = unary_out.reshape(unary_out.shape[0], -1, 1)

        pairwise_out = pairwise(img)['out']
        A = pairwise_out.reshape(pairwise_out.shape[0], pairwise_out.shape[1] // n_classes, -1)

        x = crf(A, B)
        res = x.reshape(unary_out.shape)
        res = F.interpolate(res, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)

        loss = criterion(res, seg)
        loss.backward()
        opt.step()
        sch.step()

        train_pred = torch.argmax(res, dim=1)
        train_iou = calc_iou(train_pred, seg)
        train_acc = calc_acc(train_pred, seg)
        writer.add_scalar('train/loss', loss.item(), iter_n)
        writer.add_scalar('train/IoU', train_iou, iter_n)
        writer.add_scalar('train/acc', train_acc, iter_n)
    
        if iter_n % 100 == 0:
            writer.add_image('images/img', vutils.make_grid(img[0].detach(), normalize=True, scale_each=True), iter_n)
            writer.add_image('images/gt', train_unary.pred_to_img(seg[0].detach()), iter_n, dataformats='HWC')
            writer.add_image('images/pred', train_unary.pred_to_img(train_pred[0].detach()), iter_n, dataformats='HWC')
            print(iter_n)

        iter_n += 1
    
    return unary, pairwise, iter_n


def evaluate_final(unary, pairwise, val_loader, iter_n, n_classes=21):
    criterion = nn.CrossEntropyLoss()
    solver = ConjugateGradients(shift=0.2)
    crf = dense_gaussian_crf(solver)

    unary.eval()
    pairwise.eval()
    acc = 0
    iou = 0
    count = 0
    loss = 0
    for img, seg in val_loader:
        img = img.to(device)
        seg = seg.long().to(device)

        unary_out = unary(img)['out']
        B = unary_out.reshape(unary_out.shape[0], -1, 1)

        pairwise_out = pairwise(img)['out']
        A = pairwise_out.reshape(pairwise_out.shape[0], pairwise_out.shape[1] // n_classes, -1)

        x = crf(A, B)
        res = x.reshape(unary_out.shape)
        res = F.interpolate(res, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)

        loss += criterion(res, seg).item()
        pred = torch.argmax(res, dim=1)
        iou += calc_iou(pred, seg)
        acc += calc_acc(pred, seg)

        if count == 0:
            writer.add_image('images/val/img', vutils.make_grid(img[0].detach(), normalize=True, scale_each=True), iter_n)
            writer.add_image('images/val/gt', train_unary.pred_to_img(seg[0].detach()), iter_n, dataformats='HWC')
            writer.add_image('images/val/pred', train_unary.pred_to_img(pred[0].detach()), iter_n, dataformats='HWC')

        count += 1
        if count == 100:
            break

    return loss / count, acc / count, iou / count



def train_final(train_loader, val_loader, unary, pairwise, opt, sch, n_epochs, start_epoch=0, iter_n=0):
    since = time.time()
    if not os.path.exists('models'):
        os.makedirs('models')

    if not os.path.exists('models/pairwise'):
        os.makedirs('models/pairwise')

    if not os.path.exists('models/unary_crf'):
        os.makedirs('models/unary_crf')

    for i in range(n_epochs):
        unary, pairwise, iter_n = train_epoch_final(train_loader, unary, pairwise, opt, sch, iter_n)
        cur_time = time.time()
        print('Executed %d seconds' %(cur_time - since))

        torch.save(unary.state_dict(), 'models/unary_crf/unary_crf__' + str(start_epoch + i) + '.pth')
        torch.save(pairwise.state_dict(), 'models/pairwise/pairwise__' + str(start_epoch + i) + '.pth')
        val_loss, val_acc, val_iou = evaluate_final(unary, pairwise, val_loader, iter_n)
        writer.add_scalar('val/loss', val_loss, iter_n)
        writer.add_scalar('val/IoU', val_iou, iter_n)
        writer.add_scalar('val/acc', val_acc, iter_n)
    
    return unary, pairwise, iter_n
    
