import torch
from attack import IPGD
from utils import AvgMeter
from tqdm import tqdm
import time
import numpy as np
import argparse
import json
import os
def evalRoboustness(net, batch_generator):

    defense_accs = AvgMeter()

    epsilons = [4, 8, 12, 16, 20, 24]
    nb_iters = [40, 80, 120]
    Attacks = []
    for e in epsilons:
        e = e / 255.0
        for nb in nb_iters:
            Attacks.append(IPGD(e, e//2, nb))

    net.eval()
    pbar = tqdm(batch_generator)

    for data, label in pbar:
        data = data.cuda()
        label = label.cuda()

        choices = np.random.randint(low = 0, high = 17, size = 4)
        for c in choices:
            defense_accs.update(Attacks[c].get_batch_accuracy(net, data, label))

        pbar.set_description('Evulating Roboustness')

    return defense_accs.mean

def evalRoboustness2(net, batch_generator):

    net.eval()
    epss = [4, 6, 8, 10, 12, 16]
    iters = [20, 20, 20, 20, 20, 20]
    acc = []

    for e, i in zip(epss, iters):
        e = e / 255.0
        acc.append(evalGivenEps(net, batch_generator, e, i))
    return acc
def evalGivenEps(net, batch_generator, eps, nb_iter):
    defense_accs = AvgMeter()
    net.eval()
    attack = IPGD(eps, eps / 2.0, nb_iter)

    pbar = tqdm(batch_generator)

    for data, label in pbar:
        data = data.cuda()
        label = label.cuda()

        defense_accs.update(attack.get_batch_accuracy(net, data, label))
        pbar.set_description('Evulating Roboustness')

    return defense_accs.mean

def code_test(model_path):

    from base_model.cifar_resnet18 import cifar_resnet18
    from dataset import create_test_dataset

    ds_val = create_test_dataset()

    #model_path = '../exps/exp0/checkpoint.pth.tar'

    net = cifar_resnet18()

    if model_path is not None:

        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['state_dict'])

    net.cuda()

    epoch = next(ds_val.epoch_generator())
    roboustness = evalRoboustness(net, epoch)

    print(roboustness)
    if model_path is not None:
        s = model_path.split('/')[-1]
        val_res_path = os.path.join(model_path[:-18], 'r_results.txt')
    with open(val_res_path, 'a') as f:
        f.write('\n')
        json.dump({'Roboustness': roboustness}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type = str, default = None)
    args = parser.parse_args()

    code_test(args.model_path)