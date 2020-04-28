import torch.nn as nn
from torch.backends import cudnn

from utils.models import *

#TODO: add attentio params to other nets.

def init_net(dataset, net_name, device, config):    
    num_classes = 100 if dataset == 'cifar100' else 10
    if net_name == 'wide_resnet_28_10':
        if 'quantize' in config.keys():
            if config['quantize']:
                net = WideResNet_28_10_Quantized(num_classes = num_classes)
            else:
                net = WideResNet_28_10(num_classes = num_classes, 
                                       activation  = config['activation'])
    elif net_name.startswith('efficientnet'):
        net = EfficientNetBuilder(net_name, num_classes = num_classes)
    elif net_name =='densenet100':
        net = densenet_micronet(depth           = 100, 
                                num_classes     = 100, 
                                growthRate      = 12, 
                                compressionRate = 2,
                                activation      = config['activation'],
                                attention       = config['self_attention'],
                                sym             = config['attention_sym'])
    elif net_name == 'densenet172':
        net = densenet_micronet(depth           = 172, 
                                num_classes     = 100, 
                                growthRate      = 30, 
                                compressionRate = 2,
                                activation      = config['activation'],
                                attention       = config['self_attention'],
                                sym             = config['attention_sym'])
    else:
        if 'quantize' in config.keys():
            if config['quantize']:
                net = ResNet_Quantized(net, num_classes = num_classes)
            else:
                net = ResNet(net_name, 
                             num_classes = num_classes,
                             activation  = config['activation'],
                             attention   = config['self_attention'],
                             sym         = config['attention_sym'])
    net = net.to(device)
    if device == 'cuda:0':
        net = nn.DataParallel(net)
        cudnn.benchmark = True
    return net