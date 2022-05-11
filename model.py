import os
import torch
from torch import nn
from torch.nn import functional as F

class BasicBlock(nn.Module): # 얕은 구조에서 사용. Resnet 18, 34
    mul = 1 # Block 내에서 출력 채널 수 증가 X
    def __init__(self,in_ch,out_ch,stride=1):
        super(BasicBlock,self).__init__()
        # stride를 통해 이미지 크기 조정
        # 한 층의 첫 번째 블록의 시작에서 다운 샘플 (첫 번째 층 제외)
        self.conv1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,
                               kernel_size=(3,3),stride=stride,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU()
        # 이미지 크기 유지, 채널 수 유지
        self.conv2 = nn.Conv2d(in_channels=out_ch,out_channels=out_ch,
                               kernel_size=(3,3),stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1: # stride가 1이 아니면 합 연산이 불가하므로 모양 맞추기
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=out_ch,
                                                    kernel_size=1,stride=stride),
                                          nn.BatchNorm2d(num_features=out_ch))


    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BottleNeck(nn.Module):
    mul = 4
    def __init__(self,in_ch,out_ch,stride=1):
        super(BottleNeck,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch,out_channels=out_ch,
                               kernel_size=1,stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_ch)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_ch,out_channels=out_ch,
                               kernel_size=(3,3),stride=1,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=out_ch)

        self.conv3 = nn.Conv2d(in_channels=out_ch,out_channels=out_ch*self.mul,
                               kernel_size=1,stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=out_ch*self.mul)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_ch != out_ch * self.mul: # stride가 1이 아니면 합 연산이 불가하므로 모양 맞추기
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=out_ch*self.mul,
                                                    kernel_size=1,stride=stride),
                                          nn.BatchNorm2d(num_features=out_ch*self.mul))

    def forward(self,x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10):
        # ImageNet : num_classes = 1000
        # Cifar-10 : num_classes = 10
        super(ResNet,self).__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        if self.num_classes == 1000:
            self.inplanes = 64
            # ImageNet : 224x224x3 -> 112x112x64
            # Cifar-10 : 32x32x3 -> 16x16x64
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.inplanes,
                                kernel_size=(7,7),stride=2,padding=3)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU()

            self.pool1 = nn.MaxPool2d((3,3),stride=2,padding=1)
            # ImageNet : 112x112x64 -> 56x56x64
            # Cifar-10 : 16x16x64 -> 8x8x64

            self.layer1 = self.make_layers(64,num_blocks[0],stride=1)
            # ImageNet : 56x56x64 -> 56x56x64(or 256)
            # Cifar-10 : 8x8x64 -> 8x8x64(or 256)
            self.layer2 = self.make_layers(128,num_blocks[1],stride=2)
            # ImageNet : 56x56x64(or 256) -> 28x28x128(or 512)
            # Cifar-10 : 8x8x64(or 256) -> 4x4x128(or 512)
            self.layer3 = self.make_layers(256,num_blocks[2],stride=2)
            # ImageNet : 28x28x128(or 512) -> 14x14x256(or 1024)
            # Cifar-10 : 4x4x128(or 512) -> 2x2x256(or 1024)
            self.layer4 = self.make_layers(512,num_blocks[3],stride=2)
            # ImageNet : 14x14x256(or 1024) -> 7x7x512(or 2048)
            # Cifar-10 : 2x2x256(or 1024) -> 1x1x512(or 2048)

            self.avgpool = nn.AdaptiveAvgPool2d(1)
            # ImageNet : 7x7x512(or 2048) -> 1x1x512(or 2048)
            # Cifar-10 : 1x1x512(or 2048) -> 1x1x512(or 2048)

            self.flatten = nn.Flatten()
            # 1x1x512 -> (512,) or (2048,)

            self.fc = nn.Linear(512*self.block.mul,self.num_classes)
        elif num_classes == 10: # Cifar-10
            # Conv3x3 + bn +relu
            # 2n (Residual Block n개, 16 -> 16)
            # 2n (Residual Block n개, 16 -> 32)
            # 2n (Residual Block n개, 32 -> 64)
            # Global Average + Fully Connected
            # 2 + 6n개의 층
            # n:3,5,7,9,18
            # 20,32,44,56,110
            self.inplanes = 16
            # 32x32x3 -> 32x32x16
            self.conv1 = nn.Conv2d(in_channels=3,out_channels=self.inplanes,
                                kernel_size=(3,3),stride=1,padding=1)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU()

            self.layer1 = self.make_layers(16,num_blocks[0],stride=1)
            # 32x32x16 -> 16x16x16
            self.layer2 = self.make_layers(32,num_blocks[1],stride=2)
            # 16x16x16 -> 8x8x32
            self.layer3 = self.make_layers(64,num_blocks[2],stride=2)
            # 8x8x32 -> 4x4x64
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            # 4x4x64 -> 1x1x64
            self.flatten = nn.Flatten()
            # 1x1x64 -> (64,)

            self.fc = nn.Linear(64*self.block.mul,self.num_classes)

    def make_layers(self,out_c,num_block,stride=1):
        strides = [stride] + [1] * (num_block-1)
        blocks = []
        for i in range(num_block):
            blocks.append(self.block(self.inplanes,out_c,strides[i]))
            self.inplanes = self.block.mul * out_c
        return nn.Sequential(*blocks)

    def forward(self,x):
        if self.num_classes == 1000:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x)
        elif self.num_classes == 10:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = self.flatten(x)
            x = self.fc(x)
        out = F.softmax(x)
        return out

# ImageNet
def ResNet18():
    return ResNet(BasicBlock,[2,2,2,2]) # 2*2*4+2 = 18

def ResNet34():
    return ResNet(BasicBlock,[3,4,6,3]) # 2*(3+4+6+3)+2 = 34

def ResNet50():
    return ResNet(BottleNeck,[3,4,6,3]) # 3*(3+4+6+3)+2 = 50

def ResNet101():
    return ResNet(BottleNeck,[3,4,23,3]) # 3*(3+4+23+3)+2 = 101

def ResNet152():
    return ResNet(BottleNeck,[3,8,36,3]) # 3*(3+8+36+3)+2 = 152

# Cifar-10
def ResNet20():
    return ResNet(BasicBlock,[3,3,3]) # 2+6*3 = 20

def ResNet32():
    return ResNet(BasicBlock,[5,5,5]) # 2+6*5 = 32

def ResNet44():
    return ResNet(BasicBlock,[7,7,7]) # 2+6*7 = 44

def ResNet56():
    return ResNet(BasicBlock,[9,9,9]) # 2*6*9 = 56

def ResNet110():
    return ResNet(BasicBlock,[18,18,18]) # 2*6*18 = 110


# 모델 저장 함수
def save(ckpt_dir,net,optim,epoch):
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},
               os.path.join(ckpt_dir,f"model_epoch{epoch}.pth"))

# 모델 로드 함수
def load(ckpt_dir,net,optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net,optim,epoch
    ckpt_list = os.listdir(ckpt_dir)
    if ckpt_list == []:
        epoch = 0
        return net,optim,epoch
    ckpt_list.sort(key=lambda f:int(''.join(filter(str.isdigit,f))))

    dict_model = torch.load(os.path.join(ckpt_dir,ckpt_list[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(''.join(filter(str.isdigit,ckpt_list[-1])))
    return net,optim,epoch