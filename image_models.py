import numpy as np
import matplotlib.pyplot as plt
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, utils

def get_single_layer():
    net = nn.Sequential()
    net.add(nn.Dense(10))
    net.initialize(init.Normal(sigma=0.01))
    return net

def get_LeNet(pooling='avg', activation='sigmoid', batch_norm=False):
    net = nn.Sequential()
    
    if pooling == "avg":
        pool_layer = nn.AvgPool2D(pool_size=2, strides=2)
    elif pooling == "max":
        pool_layer = nn.MaxPool2D(pool_size=2, strides=2)

    if batch_norm:
        net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2),
                nn.BatchNorm(),
                nn.Activation(activation),
                pool_layer,
                nn.Conv2D(channels=16, kernel_size=5),
                nn.BatchNorm(),
                nn.Activation(activation),
                pool_layer,
                nn.Dense(120),
                nn.BatchNorm(),
                nn.Activation(activation),
                nn.Dense(84),
                nn.BatchNorm(),
                nn.Activation(activation),
                nn.Dense(10))
    else:
        net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation=activation),
                pool_layer,
                nn.Conv2D(channels=16, kernel_size=5, activation=activation),
                pool_layer,
                nn.Dense(120, activation=activation),
                nn.Dense(84, activation=activation),
                nn.Dense(10))
    return net

def get_AlexNet(pooling='avg', activation='relu', batch_norm=False):
    net = nn.Sequential()

    if pooling == "avg":
        pool_layer = nn.AvgPool2D(pool_size=3, strides=2)
    elif pooling == "max":
        pool_layer = nn.MaxPool2D(pool_size=3, strides=2)

    if batch_norm:
        net.add(nn.Conv2D(96, kernel_size=11, strides=4),
                nn.BatchNorm(),
                nn.Activation(activation),
                pool_layer,
                nn.Conv2D(256, kernel_size=5, padding=2), 
                nn.BatchNorm(),
                nn.Activation(activation),
                pool_layer,
                nn.Conv2D(384, kernel_size=3, padding=1),
                nn.BatchNorm(),
                nn.Activation(activation),
                nn.Conv2D(384, kernel_size=3, padding=1),
                nn.BatchNorm(),
                nn.Activation(activation),
                nn.Conv2D(256, kernel_size=3, padding=1),
                nn.BatchNorm(),
                nn.Activation(activation),
                pool_layer,
                nn.Dense(4096),
                nn.BatchNorm(),
                nn.Activation(activation), 
                nn.Dropout(0.5),
                nn.Dense(4096),
                nn.BatchNorm(),
                nn.Activation(activation), 
                nn.Dropout(0.5),
                nn.Dense(10))
    else:
        net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation=activation),
                pool_layer,
                nn.Conv2D(256, kernel_size=5, padding=2, activation=activation),
                pool_layer,
                nn.Conv2D(384, kernel_size=3, padding=1, activation=activation),
                nn.Conv2D(384, kernel_size=3, padding=1, activation=activation),
                nn.Conv2D(256, kernel_size=3, padding=1, activation=activation),
                pool_layer,
                nn.Dense(4096, activation=activation), 
                nn.Dropout(0.5),
                nn.Dense(4096, activation=activation), 
                nn.Dropout(0.5),
                nn.Dense(10))
    return net

def get_VGG_11(activation='relu'):
    net = nn.Sequential()
    # The convolutional layer part
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # The fully connected layer part
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

class Residual(nn.HybridBlock):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)


def resnet18(num_classes):
    net = nn.HybridSequential()
    net.add(nn.Conv2D(64, kernel_size=3, strides=1, padding=1), 
            nn.BatchNorm(), 
            nn.Activation('relu'))

    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = nn.HybridSequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                blk.add(Residual(num_channels))
            return blk

    net.add(resnet_block(64, 2, first_block=True),
            resnet_block(128, 2),
            resnet_block(256, 2),
            resnet_block(512, 2))
    net.add(nn.GlobalAvgPool2D(), nn.Dense(num_classes))
    return net




