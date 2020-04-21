from torchvision import transforms
from torchvision.datasets import CIFAR10,CIFAR100
from torch.utils.data import DataLoader

from PIL import Image

from utils.augment.autoaugment import CIFAR10Policy
from utils.augment.cutout import Cutout


from config import dataset

#from utils.FastAutoAugment.data import get_dataloaders 
from utils.fastaugmentations import *
from utils.archive import arsaug_policy, autoaug_policy, autoaug_paper_cifar10, fa_reduced_cifar10

###########################################  TRANSFORMS  ###########################################

def get_transforms(dataloader_config):
    if dataset=='cifar10':
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    else: #cifar100
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    normalize = [transforms.Normalize(mean, std)]
    to_tensor = [transforms.ToTensor()]
    if dataloader_config['resize']:
        resize = [transforms.Resize((224,224), interpolation=Image.BICUBIC)]
    else:
        resize = []
    aug = [transforms.RandomCrop(32, padding=4), 
           transforms.RandomHorizontalFlip()]
    if dataloader_config['data_aug']:
        aug += [CIFAR10Policy()]
    if dataloader_config['use_cutout']:
        aug += [Cutout(n_holes=dataloader_config['n_holes'],
                       length =dataloader_config['length'])]
    transform_train = transforms.Compose(aug + resize + to_tensor + normalize)
    transform_test  = transforms.Compose(resize + to_tensor + normalize)
    
    if dataloader_config['use_fastaugm']:
        transform_train.transforms.insert(0, Augmentation(fa_reduced_cifar10()))

    return transform_train, transform_test


############################################  DATASET  #############################################

def get_datasets(dataloader_config, transform_train, transform_test):
    if dataset=='cifar10':
        trainset = CIFAR10(dataloader_config['rootdir'],
                           download=dataloader_config['download'],
                           train=True,
                           transform=transform_train)
        testset  = CIFAR10(dataloader_config['rootdir'],
                           download=dataloader_config['download'],
                           train=False,
                           transform=transform_test)
    else:
        trainset = CIFAR100(dataloader_config['rootdir'],
                            download=dataloader_config['download'],
                            train=True,
                            transform=transform_train)
        testset  = CIFAR100(dataloader_config['rootdir'],
                            download=dataloader_config['download'],
                            train=False,
                            transform=transform_test)
    return trainset, testset


###########################################  DATALOADER  ###########################################

def get_dataloaders(dataloader_config):
    transform_train, transform_test = get_transforms(dataloader_config)
    trainset, testset = get_datasets(dataloader_config, transform_train, transform_test)
    trainloader = DataLoader(trainset,
                             batch_size=dataloader_config['batch_size'],
                             shuffle=True,
                             num_workers=dataloader_config['nb_workers'])
    testloader  = DataLoader(testset, 
                             batch_size=dataloader_config['batch_size'], 
                             shuffle=False,
                             num_workers=dataloader_config['nb_workers'],)
    
    
    return trainloader, testloader




### FAST AUTO
class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img

