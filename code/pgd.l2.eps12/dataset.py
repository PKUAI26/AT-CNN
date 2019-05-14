from __future__ import print_function
import os
import os.path
import errno
import numpy as np
import sys
import cv2
from PIL import Image,  ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
from torchvision.datasets import ImageFolder
import torch
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


class PatchTransform(object):
    def __init__(self, k = 2):
        self.k = k

    def __call__(self, xtensor:torch.Tensor):
        '''
        X: torch.Tensor of shape(c, h, w)   h % self.k == 0
        :param xtensor:
        :return:
        '''
        patches = []
        c, h, w = xtensor.size()
        dh = h // self.k
        dw = w // self.k

        #print(dh, dw)
        sh = 0
        for i in range(h // dh):
            eh = sh + dh
            eh = min(eh, h)
            sw = 0
            for j in range(w // dw):
                ew = sw + dw
                ew = min(ew, w)
                patches.append(xtensor[:, sh:eh, sw:ew])

                #print(sh, eh, sw, ew)
                sw = ew
            sh = eh

        random.shuffle(patches)

        start = 0
        imgs = []
        for i in range(self.k):
            end = start + self.k
            imgs.append(torch.cat(patches[start:end], dim = 1))
            start = end
        img = torch.cat(imgs, dim = 2)
        return img


class SaturationTrasform(object):
    '''
    for each pixel v: v' = sign(2v - 1) * |2v - 1|^{2/p}  * 0.5 + 0.5
    then clip -> (0, 1)
    '''

    def __init__(self, saturation_level = 2.0):
        self.p = saturation_level

    def __call__(self, img):

        ones = torch.ones_like(img)
        #print(img.size(), torch.max(img), torch.min(img))
        ret_img = torch.sign(2 * img - ones) * torch.pow( torch.abs(2 * img - ones), 2.0/self.p)

        ret_img =  ret_img * 0.5 + ones * 0.5

        ret_img = torch.clamp(ret_img,0,1)

        return ret_img


class EdgeTransform(object):
    '''
    :param object:
    :return:
    '''
    def __init__(self,):
        pass
    def __call__(self, img):
        img = img.filter( ImageFilter.FIND_EDGES)
        return img

class BrightnessTransform(object):
    '''
    :param object:
    :return:
    '''
    def __init__(self, bright = 1.0):
        self.b = bright
    def __call__(self, img):
        img = ImageEnhance.Brightness(img).enhance(self.b)
        return img


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    #print(classes)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def loadPILImage(path):
    trans_img = Image.open(path).convert('RGB')
    return trans_img


def loadCVImage(path):
    img = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
    #trans_img = cv2.cvtColor(img, cv.CV_BGR2RGB)
    trans_img = cv2.cvtColor(img, cv2.CV_BGR2RGB)
    return Image.fromarray(trans_img.swapaxes(0, 2).swapaxes(1, 2).astype('uint8'), 'RGB')



def create_train_dataset(batch_size = 64, root = './'):
    print('Creating Training dataset...')
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    testset = ImageFolder('../../data/256_ObjectCategories', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    return testloader



def create_test_dataset(batch_size = 64, root = './'):
    print('Creating validation dataset...')
    transform_test = transforms.Compose([
        transforms.Resize(256),
        #transforms.RandomCrop(224),
        transforms.CenterCrop(224),
        # transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = ImageFolder('../../data/val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    return testloader

def create_style_test_dataset(batch_size = 64, root = './'):
    print('Creating validation dataset...')
    transform_test = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = StyleTinyImageNet200(root, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    return testloader

def create_saturation_test_dataset(batch_size = 64, root = './', saturation_level = 4.0):
    print('Creating validation dataset...')
    transform_test = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        SaturationTrasform(saturation_level),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = TinyImageNet200(root, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    return testloader


def create_brighness_test_dataset(batch_size = 64, root = './', bright_level = 1.0):
    print('Creating validation dataset...')
    transform_test = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(0.5),
        BrightnessTransform(bright_level),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = TinyImageNet200(root, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    return testloader

def create_patch_test_dataset(batch_size = 64, root = './', k = 2):
    print('Creating validation dataset...')
    transform_test = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        PatchTransform(k),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = TinyImageNet200(root, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    return testloader


def create_edge_test_dataset(batch_size = 64, root = './', saturation_level = 4.0):
    print('Creating validation dataset...')
    transform_test = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(32, padding=4),
        #transforms.RandomHorizontalFlip(0.5),
        EdgeTransform(),
        transforms.ToTensor(),
        #SaturationTrasform(saturation_level),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    testset = TinyImageNet200(root, train=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=4)
    return testloader

def test():
    #dataset = TinyImageNet200('./')
    dl_val = create_train_dataset()
    for img, label in dl_val:
        print(img.size())
        print(label.size())
        break




if __name__ == '__main__':
    test()
