import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from spiking_layer_ours import *
from Models import modelpool
from Preprocess import datapool
from torchvision.models.feature_extraction import create_feature_extractor
import os
import argparse
from funcs import *
import numpy as np
import time
import sys
from utils_my import replace_maxpool2d_by_avgpool2d, replace_layer_by_tdlayer, add_dimension
import calc_th_with_c as ft
from copy import deepcopy


def isActivation(name):
    if 'relu' in name.lower():
        return True
    return False


def replace_activation_by_spike(model, thresholds, thresholds1, n_steps, counter=0):
    thresholds_new = deepcopy(thresholds)
    thresholds_new1 = deepcopy(thresholds1)

    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], counter, thresholds_new = replace_activation_by_spike(module, thresholds_new, thresholds_new1, n_steps, counter)
        if isActivation(module.__class__.__name__.lower()):
            thresholds_new[counter, n_steps:] = thresholds_new1[counter, 1] / n_steps  # thresholds_out_sum/n_steps# thresholds1[counter,1] / n_steps
            thresholds_new[counter, :n_steps] = thresholds_new1[counter, 0] / n_steps  # thresholds_inner_sum/n_steps#thresholds1[counter,0] / n_steps
            model._modules[name] = SPIKE_layer(thresholds_new[counter, n_steps:], thresholds_new[counter, 0:n_steps])
            counter += 1
    return model, counter, thresholds_new


def ann_to_snn(model, thresholds, thresholds1, n_steps):
    model, counter, thresholds_new = replace_activation_by_spike(model, thresholds, thresholds1, n_steps)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_layer_by_tdlayer(model)
    return model, thresholds_new


def test_snn(model, test_loader, n_steps, criterion, device):
    model.eval()
    with torch.no_grad():
        print(n_steps)
        correct = 0
        total = 0
        loss = 0
        for images, labels in test_loader:
            images = add_dimension(images, n_steps)
            images = images.to(device)
            labels = labels.to(device)

            outputs = 0
            outputs = model(images, L=0, t=n_steps)
            outputs = torch.sum(outputs, 1)
            _, predicted = torch.max(outputs.data/n_steps, 1)
            loss += criterion(outputs/n_steps, labels).item()*images.shape[0]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # print('Accuracy of the snn network on the %d test images: %f, %d are correct'%(total,100 * correct / total,correct))
        test_loss = loss/total
        test_acc = 100 * correct / total
    return test_loss, test_acc


def train_snn(train_dataloader, test_loader, model, n_steps, epochs, optimizer,
              scheduler, device, loss_fn, args, savename,
              save_resume_ckpt=False, add_distill_loss=False, ann_model=None,
              alpha=0.5):
    model.to(device)
    best_epoch = 0
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        length = 0
        total = 0
        correct = 0

        for img, label in train_dataloader:
            if add_distill_loss:
                img_ori = deepcopy(img).to(device)
            img = add_dimension(img, n_steps)
            img = img.to(device)

            labels = label.to(device)
            outputs = model(img, L=0, t=n_steps) 
            outputs = torch.mean(outputs, 1)
            optimizer.zero_grad()
            loss = loss_fn(outputs, labels)

            if add_distill_loss:
                ann_outputs = ann_model(img_ori)
                temperature = 1
                teacher_prob = F.softmax(ann_outputs / temperature, dim=1)
                student_log_prob = F.log_softmax(outputs / temperature, dim=1)
                distill_loss = F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (temperature ** 2)
                alpha = 0.5
                loss = (1-alpha) * loss + alpha * distill_loss

            with torch.autograd.set_detect_anomaly(True):
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()*img.shape[0]
            length += len(label)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if total%(256*8) == 0:
            #     print('Epoch:%d, Accuracy of the snn network on the %d train images: %f, loss:%f'%(epoch,total,100 * correct / total,epoch_loss/total))
        print('Epoch:%d, Accuracy of the snn network on the %d train images: %f, loss:%f' % (epoch, total, 100 * correct / total, epoch_loss/total))
        scheduler.step()
        if (epoch+1) % 1 == 0:
            test_loss, test_acc = test_snn(model, test_loader, n_steps, loss_fn, device)
        if best_acc <= test_acc:
            if args.version == 'v1':
                torch.save(model.state_dict(),  savename + '_updated_snn1_%d.pth' % (n_steps))
            else:
                torch.save(model.state_dict(),  savename + '_updated_snn1_%d%s.pth' % (n_steps, args.version))
            best_acc = test_acc
            best_epoch = epoch
        print('Best acc: %f, found at the epoch: %d' % (best_acc, best_epoch))
    if save_resume_ckpt:
        resume_ckpt = {'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'scheduler_state_dict': scheduler.state_dict()}
        torch.save(resume_ckpt, savename + f'_updated_snn1_resume_e{epochs}_ckpt.pth')
    return model


def main():
    parser = argparse.ArgumentParser(description='PyTorch ANN-SNN Conversion')
    parser.add_argument('--t', default=300, type=int, help='T Latency length (Simulation time-steps)')
    parser.add_argument('--dataset', default='cifar10', type=str, help='Dataset name',
                        choices=['cifar10', 'cifar100', 'imagenet','tiny-imagenet','fashion'])
    parser.add_argument('--model', default='vgg16', type=str, help='Model name',
                        choices=['vgg16_imagenet','cifarnet','small','vgg16', 'resnet18', 'resnet20','alexnet','vgg16_no_bn',
                                'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_normed','resnet19',
                                'resnet18', 'resnet20', 'resnet34','resnet34_imagenet','resnet50', 'resnet101', 'resnet152'])
    parser.add_argument('--ann_checkpoint', default='saved_models/cifar10_vgg16_0.pth', type=str, help='path to the model checkpoint')

    parser.add_argument('--batchsize', default=64, type=int)
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=50, type=int)
    parser.add_argument('--resume_checkpoint', default='', type=str)
    parser.add_argument('--version', default='v1', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--constant_lr', action='store_true')
    parser.add_argument('--add_distill_loss', action='store_true')
    parser.add_argument('--alpha', default=0.5, type=float)

    parser.add_argument('--naive', default='', type=str)

    args = parser.parse_args()
    if args.version != 'v1':
        args.version = '_' + args.version
    savename = os.path.basename(args.ann_checkpoint).split('.')[0]
    
    batch_size = args.batchsize
    train_loader, test_loader = datapool(args.dataset, batch_size,
                                         num_workers=2, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    device = torch.device(args.device)
    n_steps = args.t
    model = modelpool(args.model, args.dataset)
    model.load_state_dict(torch.load(args.ann_checkpoint))
    num_relu = str(model).count('ReLU')
    naive = args.naive  # '_naive'#_naive
    thresholds = torch.zeros(num_relu, 2*n_steps)
    thresholds1 = torch.Tensor(np.load('%s_threshold_all_noaug%s%d.npy' % (savename, naive, 1)))
    ann_to_snn(model, thresholds, thresholds1, n_steps)
    if n_steps > 1:
        model.load_state_dict(torch.load(savename + '%s_updated_snn1_%d.pth' % (naive, n_steps-1)))
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    para1, para2, para3 = regular_set(model)
    optimizer = torch.optim.SGD([
                                {'params': para1, 'weight_decay': args.wd},
                                {'params': para2, 'weight_decay': args.wd},
                                {'params': para3, 'weight_decay': args.wd}
                                ],
                                lr=args.lr, momentum=0.9)
    if args.constant_lr:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                                    gamma=1)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs, verbose=True)
    if args.resume_checkpoint and args.start_epoch > 0:
        resume_checkpoint = torch.load(args.resume_checkpoint)
        model.load_state_dict(resume_checkpoint['model_state_dict'])
        optimizer.load_state_dict(resume_checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(resume_checkpoint['scheduler_state_dict'])
    
    if args.add_distill_loss:
        ann_model = modelpool(args.model, args.dataset)
        ann_model.load_state_dict(torch.load(args.ann_checkpoint))
        ann_model.to(device)
    else:
        ann_model = None
    

    print("Initial SNN accuracy, before training.....")
    init_loss, init_acc = test_snn(model, test_loader, n_steps, criterion,
                                   device)
    print("Initial accuracy: ", init_acc)

    running_epochs = args.end_epoch - args.start_epoch
    save_resume_ckpt = True if args.end_epoch < args.epochs else False
    model = train_snn(train_loader, test_loader, model, n_steps,
                      running_epochs, optimizer, scheduler, device,
                      criterion, args, savename, save_resume_ckpt,
                      args.add_distill_loss, ann_model, args.alpha)
    print("Final SNN accuracy, after training.......")
    test_snn(model, test_loader, n_steps, criterion, device)


if __name__ == '__main__':
    main()
