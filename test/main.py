import torch
import json
import numpy as np

from dataset import create_train_dataset, create_test_dataset
from utils import MultiStageLearningRatePolicy, save_args
import torch.optim as optim
import time
from tensorboardX import SummaryWriter
import argparse
from utils import save_checkpoint
from utils import TrainClock, AvgMeter
from attack import IPGD
import torch.nn as nn
import os
from adv_train import adversairal_train_one_epoch, adversarial_val
from collections import OrderedDict
import torchvision.models as models
from adv_test import evalRoboustness2 as evalRoboustness
parser = argparse.ArgumentParser()
parser.add_argument('--weight_decay', default=5e-4, type = float, help='weight decay (default: 5e-4)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (if has resume, this is not needed')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--exp', default=None, type = str, help = 'the name of this experiment')

parser.add_argument('--no_adv', action = 'store_true', default=True, help = 'if True, no adversarial training was used!')
parser.add_argument('--adv_freq', type = int, default=1, help = 'The frequencies of training one batch of adversarial examples')
parser.add_argument('--eps', default=8, type = int, help = 'the maximum boundary of adversarial perturbations')
parser.add_argument('--iter', default=20, type = int, help = 'the number of iterations take to generate adversarial examples for using IPGD')
parser.add_argument('-d', type = int, default=1, help = 'Which gpu to use')
#parser.add_argument('--wide', default = 2, type = int, help = 'using wider resnet18')
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.d))
if args.exp is None:
    cur_dir = os.path.realpath('./')
    args.exp = cur_dir.split(os.path.sep)[-1]
log_dir = os.path.join('../../logs', args.exp)
exp_dir = os.path.join('../../exps', args.exp)
train_res_path = os.path.join(exp_dir, 'train_results.txt')
val_res_path = os.path.join(exp_dir, 'val_results.txt')
final_res_path = os.path.join(exp_dir, 'final_results.txt')
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

save_args(args, exp_dir)
writer = SummaryWriter(log_dir)

clock = TrainClock()

learning_rate_policy = [[5, 0.01],
                        [3, 0.001],
                        [2, 0.0001]
                        ]
get_learing_rate = MultiStageLearningRatePolicy(learning_rate_policy)

def adjust_learning_rate(optimizer, epoch):
    #global get_lea
    lr = get_learing_rate(epoch)
    for param_group in optimizer.param_groups:

        param_group['lr'] = lr

torch.backends.cudnn.benchmark = True
ds_train = create_train_dataset(args.batch_size)

ds_val = create_test_dataset(args.batch_size)

net = models.resnet18(pretrained = True)
#net.avgpool = nn.AdaptiveAvgPool2d(1)
#net.fc.out_features = 200
net.fc = nn.Linear(512, 257)

net.to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.SGD(net.parameters(), lr = get_learing_rate(0), momentum = 0.9, weight_decay=args.weight_decay)

args.eps = args.eps / 255.0
PgdAttack = IPGD(eps = args.eps, sigma = args.eps / 2.0, nb_iter = args.iter, norm = np.inf,
                 DEVICE = DEVICE)


best_prec = 0.0
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        check_point = torch.load(args.resume)
        args.start_epoch = check_point['epoch']
        net.load_state_dict(check_point['state_dict'])
        best_prec = check_point['best_prec']

        print('Modeled loaded from {} with metrics:'.format(args.resume))
        print(results)
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

clock.epoch = args.start_epoch
#for epoch in ds_train.epoch_generator():
while True:
    if clock.epoch > args.epochs:
        break
    adjust_learning_rate(optimizer, clock.epoch)

    Trainresults = adversairal_train_one_epoch(net, optimizer, ds_train, criterion, PgdAttack, clock,
                                               attack_freq = args.adv_freq, use_adv = not args.no_adv,
                                               DEVICE = DEVICE)
    Trainresults['epoch'] = clock.epoch
    with open(train_res_path, 'a') as f:
        json.dump(Trainresults, f)

    torch.cuda.empty_cache()

    #val_epoch = next(ds_val.epoch_generator())
    Valresults = adversarial_val(net, ds_val, criterion, PgdAttack, clock,
                                 attack_freq = args.adv_freq * 5,
                                 DEVICE = DEVICE)
    Valresults['epoch'] = clock.epoch
    torch.cuda.empty_cache()
    with open(val_res_path, 'a') as f:
        f.write('\n')
        json.dump(Valresults, f)

    prec = Valresults['clean_acc']
    if prec > best_prec:
        best_prec = prec
        save_checkpoint(
            {"epoch": clock.epoch,
             'state_dict': net.state_dict(),
             'best_prec': best_prec}, is_best=True, prefix=exp_dir)
    else:
        save_checkpoint(
            {"epoch": clock.epoch,
             'state_dict': net.state_dict(),
             'best_prec': best_prec}, is_best=False, prefix=exp_dir)
    for name, val in Trainresults.items():
        vval = Valresults[name]

        writer.add_scalars(main_tag = name, tag_scalar_dict = {
            "Train": val,
            'Val': vval},
                           global_step = clock.epoch)

    if clock.epoch >= (args.epochs // 2) and clock.epoch % 5 ==0:
        torch.save({"epoch": clock.epoch,
             'state_dict': net.state_dict()},
                   os.path.join(exp_dir, 'epoch-{}.pth'.format(clock.epoch)))
    if clock.epoch % 25 == 0 and clock.epoch > (args.epochs // 2):
        #val_epoch = next(ds_val.epoch_generator())

        roboustness = evalRoboustness(net, ds_val)
        with open(val_res_path, 'a') as f:
            f.write('\n')
            json.dump({'Roboustness': roboustness}, f)


print('Final prec: {:.2f} --- roboustness: {:.2f}'.format(Valresults['clean_acc'], Valresults['adv_acc']))

with open(final_res_path, 'a') as f:
    f.write('\n')
    json.dump({
        "prec": Valresults['clean_acc'],
        'roboustness': Valresults['adv_acc']
    }, f)
