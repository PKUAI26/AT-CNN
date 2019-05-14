import torch
import json
import numpy as np
from adv_dataset import create_train_dataset, create_test_dataset
from classifier_test import ResNet18 as cifar_resnet18
import time
import argparse
from attack import IPGD
import os
from collections import OrderedDict
from cv2 import imwrite, imread

def generate_large_adv(net, dl, AttackMethod, DEVICE, save_dir = None):
    for i, (batch_img, batch_label) in enumerate(dl):
        if i > 5:
            break
        batch_img = batch_img.to(DEVICE)
        batch_label = batch_label.to(DEVICE)
        adv_imgs = AttackMethod.attack(net, batch_img, batch_label, target = None)

        for j in range(int(batch_img.size(0))):
            adv_img = adv_imgs[j]
            adv_img = adv_img.detach().cpu().numpy()
            #print(adv_img.shape)
            adv_img = adv_img.transpose(1,2,0)
            imwrite(os.path.join(save_dir, '{}-adv.png'.format(i * batch_img.size(0) + j)), adv_img)
    print(save_dir)
    MakeVisual(result_dir=save_dir)


def MakeVisual(data_dir = './benchmark/CIFAR', result_dir = '../adv_exps/imgs/'):
    save_p = result_dir.split('/')[:-1]
    save_p = os.path.join(*save_p)
    print(save_p)
    net_name = result_dir.split('/')[-1]
    labels = np.loadtxt(os.path.join(data_dir, 'label.txt'))
    imgs = get_a_adv_set(labels, result_dir, data_dir, times=3)
    print(os.path.join(save_p, '{}.png'.format(net_name)))
    imwrite(os.path.join(save_p, '{}.png'.format(net_name)), imgs)

def get_a_adv_set(labels, result_dir = './adv_exps/imgs',
              data_dir= './benchmark/MNIST', times = 3):
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
            rimg = imread(os.path.join(data_dir, '{}.png'.format(i)))[:,:,:]
            grad = imread(os.path.join(result_dir, "{}-adv.png".format(i)))[:,:,:]*255

            #print(rimg.max(), grad.max(), rimg.min(), grad.min())
            img = np.concatenate((grad, rimg), axis= 1)
            imgs.append(img)

        rets.append(np.concatenate(imgs, axis = 0))
    ret = np.concatenate(rets, axis = 1)
    return ret

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save_path', default='../adv_exps/imgs')
    parser.add_argument('--iter', default=120, type=int,
                        help='the number of iterations take to generate adversarial examples for using IPGD')
    parser.add_argument('--eps', default=40, type=int,
                        help='the maximum boundary of adversarial perturbations')
    parser.add_argument('-d', type=int, default=0)
    args = parser.parse_args()
    DEVICE = torch.device('cuda:{}'.format(args.d))
    save_dir = os.path.join('../adv_exps/imgs/',args.resume.split('/')[-2])
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    ds_val = create_test_dataset(32)
    net = cifar_resnet18(3)
    print('loading at {}'.format(args.resume))
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['state_dict'])
    net.to(DEVICE)
    print(save_dir)
    PgdAttack = IPGD(eps=args.eps / 255.0, sigma= 1 / 255.0, nb_iter=args.iter, norm=np.inf)

    generate_large_adv(net, ds_val, PgdAttack, DEVICE, save_dir)