import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from time import time
import datetime
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *

parser = argparse.ArgumentParser(description='ResNet Model')
parser.add_argument('--layers',choices=['18','34','50','101','152','20','32','44','56','110','all'],required=True,
                    help='ImageNet : 18,34,50,101,152 Cifar-10 : 20,32,44,56,110')
parser.add_argument('--data',choices=['imagenet','cifar10'],default='cifar10')
parser.add_argument('--lr',type=float,default=0.1,help='')
parser.add_argument('--batch_size',type=int,default=128,help='')
parser.add_argument('--num_workers',type=int,default=4,help='')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = args.lr
batch_size = args.batch_size
num_workers = args.num_workers
train_size = 45000 # 45k / 5k
val_size = 5000
num_iteration = 64000
root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir,'data')

# Prepare DataLoader
train_transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])

train_dataset0 = datasets.CIFAR10(root='./data',
                                train=True,
                                download=True,
                                transform=train_transform,)

train_dataset, val_dataset = random_split(train_dataset0,[train_size,val_size])

test_dataset = datasets.CIFAR10(root='./data',
                                train=False,
                                download=True,
                                transform=test_transform)

classes = train_dataset0.classes
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers)

# Parameters
num_data_train = len(train_dataset)
num_data_val = len(val_dataset)
num_data_test = len(test_dataset)
num_batch_train = int(np.ceil(num_data_train/batch_size))
num_batch_val = int(np.ceil(num_data_val/batch_size))
num_batch_test = int(np.ceil(num_data_test/batch_size))
num_epoch =  int(np.ceil(num_iteration /num_batch_train))# 64000 iteration


if args.layers == 'all':
    layers = ['18','34','50','101','152','20','32','44','56','110']
else:
    layers = [args.layers]

for layer in layers:
    model_name = 'ResNet'+layer
    ckpt_dir = os.path.join(root_dir,'checkpoint',model_name+'_'+args.data)
    os.makedirs(ckpt_dir,exist_ok=True)
    log_dir = os.path.join(root_dir,'logs',model_name+'_'+args.data)

    # Model
    net = locals()[model_name]().to(device)
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("{} Parameters : {}".format(model_name,num_params))

    # Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optim = torch.optim.SGD(net.parameters(),lr=lr,momentum=0.9,weight_decay=1e-4)
    decay_epoch = [32000,48000]
    step_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                            decay_epoch,
                                                            gamma=0.1)

    # Tensorboard
    writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
    writer_val = SummaryWriter(log_dir=os.path.join(log_dir,'val'))
    writer_test = SummaryWriter(log_dir=os.path.join(log_dir,'test'))

    # Function
    fn_tonumpy = lambda x:x.to('cpu').detach().numpy().transpose(1,2,0)
    def fn_denorm(x,mean=(0.4914,0.4822,0.4465),std=(0.2023,0.1994,0.2010)):
        for i in range(x.shape[0]):
            x[i] = (x[i]* std[i]) + mean[i]
        return x
    def fn_diff_index(preds,labels):
        for i in range(len(preds)):
            if preds[i] != labels[i]:
                return i
        return None
    def make_figure(inputs_,preds_,labels_):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.imshow((inputs_*255).astype(np.uint8))
        ax.set_title(f"Prediction : {preds_} Label : {labels_}",size=15)
        return fig
    def train(global_step):
        # Train
        net.train()
        loss_arr = []
        acc_arr = []

        for inputs,labels in train_loader:
            global_step += 1
            inputs = inputs.to(device) # To GPU
            labels = labels.to(device) # To GPU
            outputs= net(inputs) # Forward Propagation
            # Backpropagation
            optim.zero_grad()
            loss = loss_fn(outputs,labels)
            loss.backward()
            optim.step()
            step_lr_scheduler.step() # Scheduler Increase Step
            # Metric
            loss_arr.append(loss.item())
            _, preds = torch.max(outputs.data,1)
            acc_arr.append(((preds==labels).sum().item()/labels.size(0))*100)
            # Print
            print(f"TRAIN: STEP {global_step:05d} / {total_step:05d} | LOSS {np.mean(loss_arr):.4f} | ACC {np.mean(acc_arr):.2f}%")
            # Tensorboard
            p = fn_diff_index(preds,labels)
            if p is not None:
                inputs_ = fn_tonumpy(fn_denorm(inputs[p]))
                labels_ = classes[labels[p]]
                preds_ = classes[preds[p]]
                fig = make_figure(inputs_,preds_,labels_)
                writer_train.add_figure('Pred vs Target',fig,global_step)
                writer_train.add_scalar('Loss',np.mean(loss_arr),global_step)
                writer_train.add_scalar('Error',100-np.mean(acc_arr),global_step)
                writer_train.add_scalar('Accuracy',np.mean(acc_arr),global_step)
        return global_step
    def valid(global_step):
        with torch.no_grad():
            net.eval()
            loss_arr = []
            acc_arr = []

            for inputs,labels in val_loader:
                inputs = inputs.to(device) # To GPU
                labels = labels.to(device) # To GPU
                outputs= net(inputs) # Forward Propagation
                # Backpropagation
                loss = loss_fn(outputs,labels)
                # Metric
                loss_arr.append(loss.item())
                _, preds = torch.max(outputs.data,1)
                acc_arr.append(((preds==labels).sum().item()/labels.size(0))*100)
            # Print
            print(f"VALID: STEP {global_step:05d} / {total_step:05d} | LOSS {np.mean(loss_arr):.4f} | ACC {np.mean(acc_arr):.2f}%")
            # Tensorboard
            p = fn_diff_index(preds,labels)
            if p is not None:
                inputs_ = fn_tonumpy(fn_denorm(inputs[p]))
                labels_ = classes[labels[p]]
                preds_ = classes[preds[p]]
                fig = make_figure(inputs_,preds_,labels_)
                writer_val.add_figure('Pred vs Target',fig,global_step)
                writer_val.add_scalar('Loss',np.mean(loss_arr),global_step)
                writer_val.add_scalar('Error',100-np.mean(acc_arr),global_step)
                writer_val.add_scalar('Accuracy',np.mean(acc_arr),global_step)
    def test():
        with torch.no_grad():
            net.eval()
            loss_arr = []
            acc_arr = []

            for inputs,labels in test_loader:
                inputs = inputs.to(device) # To GPU
                labels = labels.to(device) # To GPU
                outputs= net(inputs) # Forward Propagation
                # Backpropagation
                loss = loss_fn(outputs,labels)
                # Metric
                loss_arr.append(loss.item())
                _, preds = torch.max(outputs.data,1)
                acc_arr.append(((preds==labels).sum().item()/labels.size(0))*100)
            # Print
            print(f"TEST: LOSS {np.mean(loss_arr):.4f} | ACC {np.mean(acc_arr):.2f}%")
            writer_test.add_scalar('Loss',np.mean(loss_arr))
            writer_test.add_scalar('Error',100-np.mean(acc_arr))
            writer_test.add_scalar('Accuracy',np.mean(acc_arr))

    # start_epoch=0
    global_step = 0
    total_step = num_epoch * num_batch_train # 64064
    start_time = time()
    for epoch in range(1,num_epoch+1):
        global_step = train(global_step)
        valid(global_step)
    total_time = time() - start_time
    test()
    writer_train.add_text('Parameters',str(num_params))
    writer_train.add_text('Train Time',str(datetime.timedelta(seconds=total_time)))
    writer_train.close()
    writer_val.close()
    writer_test.close()
