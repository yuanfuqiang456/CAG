#coding:utf8
from torch import nn
from .BasicModule import BasicModule
import torch.nn.functional as F

class AlexNet(BasicModule):
    '''
    code from torchvision/models/alexnet.py
    结构参考 <https://arxiv.org/abs/1404.5997>
    '''
    def __init__(self, num_classes=2):
        
        super(AlexNet, self).__init__()
        
        self.model_name = 'alexnet'


        self.features = nn.Sequential(
            #特征图大小 512
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            #SyncBatchNorm(64),
            #特征图大小 127
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #特征图大小 62
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            #SyncBatchNorm(192),

            #特征图大小 62
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            #特征图大小 30
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            #SyncBatchNorm(384),
            #特征图大小 30
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            #SyncBatchNorm(256),
            #特征图大小 30
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            #SyncBatchNorm(256),
            #特征图大小 30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.addCNN = nn.Sequential(
            # 特征图大小 14
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # 特征图大小 14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 特征图大小 7
        )

        self.classifier = nn.Linear(256, num_classes)
        '''
        self.classifier1 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        '''

    def forward(self, x):
        x = self.features(x)
        x1 = self.addCNN(x)
        out = F.relu(x1, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(x1.size(0), -1)
        x = self.classifier(out)
        return x,x1
