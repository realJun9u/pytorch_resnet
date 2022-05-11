import os
import torch
import numpy as np
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import ResNet50,Resnet56,ResNet152,ResNet110,load,save

root_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(root_dir,'data')
ckpt_dir = os.path.join(root_dir,'checkpoint')
log_dir = os.path.join(root_dir,'logs')
batch_size = 128
num_epoch = 64000
train_size = 45000 # 45k / 5k
val_size = 5000

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
                                 transform=train_transform)

train_dataset, val_dataset = random_split(train_dataset0,[train_size,val_size])

test_dataset = datasets.CIFAR10(root='./data',
                                 train=False,
                                 download=True,
                                 transform=test_transform)

classes = train_dataset0.classes
train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# Model
device = "cuda" if torch.cuda.is_available() else "cpu"
net = ResNet50().to(device)

num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
# Loss Function
loss_fn = torch.nn.CrossEntropyLoss()
# Optimizer
optim = torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9,weight_decay=1e-4)
decay_epoch = [32000,48000]
step_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                         decay_epoch,
                                                         gamma=0.1)
# 저장된 최근의 체크 포인트 불러오기
net, optim, start_epoch = load(ckpt_dir,net,optim)
# Function
fn_tonumpy = lambda x:x.to('cpu').detach().numpy().transpose(0,2,3,1)
def fn_denorm(x,mean=(0.4914,0.4822,0.4465),std=(0.2023,0.1994,0.2010)):
    for i in range(x.shape[0]):
        for j in range(3):
            x[i][j] = (x[i][j] * std[j]) + mean[j]
    return x

# Parameters
num_data_train = len(train_dataset)
num_data_val = len(val_dataset)
# num_data_test = len(test_dataset)
num_batch_train = int(np.ceil(num_data_train/batch_size))
num_batch_val = int(np.ceil(num_data_val/batch_size))
# num_batch_test = int(np.ceil(num_data_test/batch_size))

# Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir,'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir,'val'))
# writer_test = SummaryWriter(log_dir=os.path.join(log_dir,'test'))

for epoch in range(start_epoch + 1,num_epoch+1):
    # Train
    net.train()
    loss_arr = []
    acc_arr = []
    
    for batch, (inputs,labels) in enumerate(train_loader,start=1):
        step_lr_scheduler.step() # Scheduler Increase Step
        
        inputs = inputs.to(device) # To GPU
        labels = labels.to(device) # To GPU
        outputs= net(inputs) # Forward Propagation
        _, preds = torch,max(outputs.data,1)
        # Backpropagation
        optim.zero_grad()
        loss = loss_fn(preds,labels)
        loss.backword()
        optim.step()
        # Metric
        loss_arr.append(loss.item())
        acc_arr.append((preds==labels).sum().item()/labels.size(0))
        # Print
        print(f"TRAIN: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_train:04d} | LOSS {np.mean(loss_arr):.4f}")
        # Tensorboard
        p = np.random.randint(batch_size)
        inputs_ = fn_tonumpy(fn_denorm(inputs[p]))
        labels_ = classes(labels[p])
        preds_ = classes(preds[p])
        writer_train.add_image('Input',inputs_,num_batch_train*(epoch-1)+batch,dataformats='NHWC')
        writer_train.add_text('Prediction',preds_,num_batch_train*(epoch-1)+batch)
        writer_train.add_text('Target',labels_,num_batch_train*(epoch-1)+batch)
    writer_train.add_scalar('Loss',np.mean(loss_arr),epoch)
    writer_train.add_scalar('Accuracy',np.mean(acc_arr),epoch)
    # Validation
    with torch.no_grad():
        net.eval()
        loss_arr = []
        acc_arr = []

        for batch, (inputs,labels) in enumerate(val_loader,start=1):
            inputs = inputs.to(device) # To GPU
            labels = labels.to(device) # To GPU
            outputs= net(inputs) # Forward Propagation
            _, preds = torch,max(outputs.data,1)
            # Backpropagation
            loss = loss_fn(preds,labels)
            # Metric
            loss_arr.append(loss.item())
            acc_arr.append((preds==labels).sum().item()/labels.size(0))
            # Print
            print(f"VALID: EPOCH {epoch:04d} / {num_epoch:04d} | BATCH {batch:04d} / {num_batch_val:04d} | LOSS {np.mean(loss_arr):.4f}")
            # Tensorboard
            p = np.random.randint(batch_size)
            inputs_ = fn_tonumpy(fn_denorm(inputs[p]))
            labels_ = classes(labels[p])
            preds_ = classes(preds[p])
            writer_val.add_image('Input',inputs_,num_batch_val*(epoch-1)+batch,dataformats='NHWC')
            writer_val.add_text('Prediction',preds_,num_batch_val*(epoch-1)+batch)
            writer_val.add_text('Target',labels_,num_batch_val*(epoch-1)+batch)
        writer_val.add_scalar('Loss',np.mean(loss_arr),epoch)
        writer_val.add_scalar('Accuracy',np.mean(acc_arr),epoch)
    
    # epoch 500 마다 저장
    if epoch % 500 == 0:
        save(ckpt_dir,net,optim,epoch)

writer_train.close()
writer_val.close()
# writer_test.close()
