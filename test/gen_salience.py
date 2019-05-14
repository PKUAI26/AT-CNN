import torch
import os
from utils import GetSmoothGrad
import cv2
from cv2 import imwrite, imread
import argparse
import torch
import numpy as np
from utils import get_a_set
import torch.nn.functional as F
import torch.nn as nn

import torchvision.models as models
import skimage.io as io
from skimage import transform
from torchvision import transforms

def clip_gradmap_to_gray(gradmap):
    '''
    :param gradmap: should be a single image!! shape of (c, h, w)
    :return: a image of the same shape.   values of pixels: (0, 255) unint8
    '''
    if isinstance(gradmap, torch.Tensor):
        gradmap = gradmap.cpu().numpy()

    assert len(gradmap.shape) == 3 or len(gradmap.shape == 2),  gradmap.shape

    gradmap = np.abs(gradmap)

    gradmap = np.sum(gradmap, axis = 0)

    max_value = np.percentile(gradmap, 99)
    #print(max_value)
    img = np.clip( (gradmap) / ( max_value), 0, 1) * 255
    img = img.astype(np.uint8)
    #print('g', img.max(), img.min(), img.shape) 0-255 (224, 224)

    return img

def GetGraySmoothGrad(net, img:torch.Tensor, label:torch.Tensor, DEVICE):
    grad_map = GetSmoothGrad(net, img, label, DEVICE, stdev_spread=0.10)
    # 3x224x224
    grad = GetSmoothGrad(net, img, label, DEVICE, stdev_spread = 0.0, num=1)

    gray_grad = clip_gradmap_to_gray(grad_map)
    grad = clip_gradmap_to_gray(grad)
    #print(gray_grad.shape)
    gray_grad = gray_grad[:, :, np.newaxis]
    grad = grad[:, :, np.newaxis]
    gray_grad = np.repeat(gray_grad, 3, axis=2) # (224, 224, 3)
    grad = np.repeat(grad, 3, axis=2)
    #print(gray_grad.shape)
    return gray_grad, grad

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image_path', type = str, help='Path of Clean image')
    #parser.add_argument('-d', type=int, default=0, help = 'Which gpu to use')
    args = parser.parse_args()

    root = './Maps/imgs'

    device = torch.device('cuda')
    #device = torch.cuda()

    model_paths = [
        '../exps/baseline.re/checkpoint.pth.tar',
        '../exps/underfit/checkpoint.pth.tar',
        '../exps/tradeoff.eps8/checkpoint.pth.tar',
        '../exps/tradeoff.eps4/checkpoint.pth.tar',
        '../exps/tradeoff.eps2/checkpoint.pth.tar',
        '../exps/tradeoff.eps1/checkpoint.pth.tar',
        '../exps/two.eps12/checkpoint.pth.tar',
             '../exps/two.eps8/checkpoint.pth.tar',
             '../exps/two.eps4/checkpoint.pth.tar',
        '../exps/fgsm.eps16/checkpoint.pth.tar',
        '../exps/fgsm.eps8/checkpoint.pth.tar',
    ]


    modelpool = []
    for model_path in model_paths:
        model = models.resnet18(False)
        model.fc = nn.Linear(512, 257)
        print('loading', model_path)
        #a = torch.load(model_path)['state_dict']
        #print(a)
        model.load_state_dict(torch.load(model_path, map_location='cpu')['state_dict'], strict=True)
        model.to(device)
        model.eval()
        modelpool.append(model)

    label_path = os.path.join(os.path.dirname(args.image_path), 'label.txt')
    labels = np.loadtxt(label_path)
    img_idx = int(args.image_path.split('/')[-1][:-4]) - 1
    label = int(labels[img_idx])
    label = torch.tensor(label)
    #label.expand(num=1)

    raw_image = cv2.imread(args.image_path)[..., ::-1]
    # print(raw_image.shape)
    # raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])(raw_image)#.unsqueeze(0)

    print(raw_image.shape, image.size(), label)
    # raw_image: 224x224x3

    AllImages = [raw_image]
    AllGrads = [raw_image]
    for net in modelpool:
        img, grad = (GetGraySmoothGrad(net, image, label, device))
        AllImages.append(img)
        AllGrads.append(grad)

    figure = np.concatenate(AllImages, axis = 1)
    Grad = np.concatenate(AllGrads, axis=1)

    print(figure.shape)

    save_path = os.path.join('./results', args.image_path.split('/')[-1])
    save_path_grad = os.path.join('./results', 'grad-'+args.image_path.split('/')[-1])
    io.imsave(save_path, figure)
    io.imsave(save_path_grad, Grad)
