import copy
import os
import numpy as np
from PIL import  Image
from cv2 import imwrite, imread
import json
import argparse
import torch
import torch.nn as nn
import math
from typing import Tuple, List, Dict
from abc import abstractmethod, abstractproperty, ABCMeta
import shutil
import skimage.io as io

def clip_gradmap(gradmap, transpose = True, togray = True):
    '''
    :param gradmap: should be a single image!! shape of (c, h, w)
    :return: a image of the same shape.   values of pixels: (0, 255) unint8
    '''
    if isinstance(gradmap, torch.Tensor):
        gradmap = deepcopy(gradmap.numpy())

    assert len(gradmap.shape) == 3 or len(gradmap.shape == 2),  gradmap.shape

    gradmap = np.abs(gradmap)

    if gradmap.shape[0] == 1:
        gradmap = gradmap[0]
    if togray and gradmap.shape[0] == 3:
        gradmap = np.sum(gradmap, axis = 0)

    max_value = np.percentile(gradmap, 99)
    #print(max_value)
    img = np.clip( (gradmap) / ( max_value), 0, 1) * 255
    img = img.astype(np.uint8)
    #print('g', img.max(), img.min())
    if transpose and not togray:
        img = np.transpose(img, (1, 2, 0))

    return img

def clip_and_save_single_img(grad_map, i = 0, save_dir = './benchmark_smooth'):
    grad_map = grad_map.detach().cpu()
    grad_map = grad_map.numpy()

    image_name = os.path.join(save_dir, '{}-smooth.png'.format(i))

    #print('img', grad_map.shape)
    #img = clip_gradmap(grad_map, togray=True)
    img = clip_gradmap(grad_map, togray=False)
    #print(img.shape)
    #imwrite(image_name, img)
    io.imsave(image_name, img)
    return img


def clip_and_save_batched_imgs(grad_maps, start_i = 0, save_dir = './benchmark_results'):
    grad_maps = grad_maps.detach().cpu()
    grad_maps = grad_maps.numpy()

    start_i = start_i * grad_maps.shape[0]
    for i, grad_map in enumerate(grad_maps):
        image_name = os.path.join(save_dir, '{}-result.png'.format(i + start_i))
        #print('img',img.shape)
        img = clip_gradmap(grad_map, togray=False)
        imwrite(image_name, img)

def GetSmoothGrad(net, img, label, DEVICE, stdev_spread = 0.15, num = 100):
    '''
    :param net: pytorch network
    :param img: a single image
    :stdev_spread: Amount of noise to add to the input, as fraction of the
                    total spread (x_max - x_min). Defaults to 15%.
    :num:  Number of samples used
    :square: If True, computes the sum of squares of gradients instead of
                 just the sum. Defaults to False.
    :return: smooth grad
    '''

    size = list(img.size())
    size = [num, ] + size
    img = img.expand((size))
    #print('daw', label)
    label = label.expand((num))
    #print(img.size())
    #print(label.size())
    net.eval()
    img = img.to(DEVICE)
    label = label.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    img.requires_grad = True

    stdev = (torch.max(img) - torch.min(img)) * stdev_spread
    stdev = stdev.expand_as(img)
    mean = torch.zeros_like(img)
    noises = torch.normal(mean, stdev)
    img = img + noises

    pred = net(img)
    loss = criterion(pred, label)
    #grad_maps = torch.autograd.grad(loss, img, create_graph=True, only_inputs=False)[0]
    grad_maps = torch.autograd.grad(loss, img, retain_graph = False, only_inputs=False)[0]
    #print(torch.norm(grad_maps, 1).item())
    grad_map = torch.sum(grad_maps, dim = 0)

    return grad_map



def get_a_set(labels, result_dir = '../SmoothRes/',
              data_dir= '../data/benchmark', times = 1):
    rets = []
    labels = list(labels)
    indexs = list(range(len(labels)))

    for t in range(times):
        sets = []
        for target in range(10):
            for j, i in enumerate(labels):
                if i == target:
                    sets.append(indexs[j])
                    labels.pop(j)
                    indexs.pop(j)
                    break
        imgs = []
        for i in sets:
            print(os.path.join(data_dir, '{}.png'.format(i)))
            os.path.join(result_dir, "{}-smooth.png".format(i))
            rimg = imread(os.path.join(data_dir, '{}.png'.format(i)))[:,:,:]
            grad = imread(os.path.join(result_dir, "{}-smooth.png".format(i)))[:,:,:]
            #print(rimg.max(), grad.max(), rimg.min(), grad.min())
            img = np.concatenate((grad, rimg), axis= 1)
            imgs.append(img)

        rets.append(np.concatenate(imgs, axis = 0))
    ret = np.concatenate(rets, axis = 1)
    return ret



class TrainClock(object):
    def __init__(self):
        self.epoch = 0
        self.minibatch = 0
        self.step = 0

    def tick(self):
        self.minibatch += 1
        self.step += 1

    def tock(self):
        self.epoch += 1

        self.minibatch = 0


class AvgMeter(object):
    name = 'No name'
    def __init__(self, name = 'No name'):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = 0
        self.mean = 0
        self.num = 0
        self.now = 0
    def update(self, mean_var, count = 1):
        if math.isnan(mean_var):
            mean_var = 1e6
            print('Avgmeter getting Nan!')
        self.now = mean_var
        self.num += count

        self.sum += mean_var * count
        self.mean = float(self.sum) / self.num


class MultiStageLearningRatePolicy(object):
    '''
    '''

    _stages = None
    def __init__(self, stages:List[Tuple[int, float]]):

        assert(len(stages) >= 1)
        self._stages = stages


    def __call__(self, cur_ep:int) -> float:
        e = 0
        for pair in self._stages:
            e += pair[0]
            if cur_ep < e:
                return pair[1]
      #  return pair[-1][1]
        return pair[-1]


def save_args(args, save_dir = None):
    if save_dir == None:
        param_path = os.path.join(args.resume, "params.json")
    else:
        param_path = os.path.join(save_dir, 'params.json')

    #logger.info("[*] MODEL dir: %s" % args.resume)
    #logger.info("[*] PARAM path: %s" % param_path)

    with open(param_path, 'w') as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)



def dump_dict(json_file_path, dic):
    with open(json_file_path, 'a') as f:
        f.write('\n')
        json.dump(dic, f)

def save_checkpoint(state, is_best, prefix = 'exp0'):
    filename='checkpoint.pth.tar'
    filepath = os.path.join(prefix, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(prefix, 'model_best.pth.tar'))
def torch_accuracy(output, target, topk = (1, )):
    '''
    param output, target: should be torch Variable
    '''
    #assert isinstance(output, torch.cuda.Tensor), 'expecting Torch Tensor'
    #assert isinstance(target, torch.Tensor), 'expecting Torch Tensor'
    #print(type(output))

    topn = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(topn, 1, True, True)
    pred = pred.t()

    is_correct = pred.eq(target.view(1, -1).expand_as(pred))

    ans = []
    for i in topk:
        is_correct_i = is_correct[:i].view(-1).float().sum(0, keepdim = True)
        ans.append(is_correct_i.mul_(100.0 / batch_size))

    return ans


class SalienceGenerator(object):
    def __init__(self, stdev_spread = 0.15, num = 32):
        self.stdev_speard = stdev_spread
        self.num = num
        self.device = torch.device('cuda:0')

    def __call__(self, net, data, label):
        grad_maps = []
        for i in range(data.size(0)):
            img = data[0]
            l = label[0]
            grad_map = GetSmoothGrad(net, img, l, self.device, self.stdev_speard, self.num)
            # gray scale
            grad_map = torch.sum(grad_map, dim = 0, keepdim = True)
            grad_l = torch.reshape(grad_map, (1, -1))
            max_normalize,_ = torch.topk(grad_l, (32*32 )// 100)
            #print(max_normalize)
            max_normalize = max_normalize[0][-1]
            grad_map = torch.abs(grad_map) /max_normalize
            grad_map = torch.clamp(grad_map, 0, max_normalize.item())
            grad_map = grad_map.detach()
            grad_maps.append(grad_map)

        grad_map = torch.stack(grad_maps)
        return grad_map

#def PairImg(p1 = '')
