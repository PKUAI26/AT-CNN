from utils import GetSmoothGrad, clip_and_save_single_img, clip_gradmap
import os
from cv2 import imwrite, imread
import argparse
import torch
import numpy as np
import torch
from utils import get_a_set
import torch.nn.functional as F
import torch.nn as nn
from dataset import create_test_dataset, create_train_dataset
import torchvision.models as models
import skimage.io as io
def GetSmoothRes(net, Data, DEVICE, save_path ='./SmoothRes/Fashion_MNIST'):
    for i, (img, label) in enumerate(zip(Data.X, Data.Y)):
        #print(i)
        #print(img.shape, label.shape)
        img = img.astype(np.float32)
        #label = label.astype(np.float32)
        img = img[np.newaxis,:]
        img = torch.tensor(img)
        #print(img.type())
        label = torch.tensor(label).type(torch.LongTensor)
        grad_map = GetSmoothGrad(net, img, label, DEVICE = DEVICE)
        grad_map = grad_map.cpu().detach().numpy()
        grad_map = clip_gradmap(grad_map)
        #print(grad_map.shape, grad_map.mean())
        save_p = os.path.join(save_path, '{}.png'.format(i))
        #print(grad_map.shape)
        imwrite(save_p, grad_map)
    print('{} imgs saved in {}'.format(i+1, save_path))


def get_result(net, dl, DEVICE, net_name = ''):
    save_bench = '../data/benchmark/'
    save_path = os.path.join('../Res/', net_name)
    labels = []
    net.eval()
    mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    std = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    mean = mean.to(DEVICE)
    std = std.to(DEVICE)
    for i, (batch_img, batch_label) in enumerate(dl):
        if i> 5:
            break
        for j in range(int(batch_img.size(0))):
            img = batch_img[j]
            label = batch_label[j]
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            #print(img.size())
            grad_map = GetSmoothGrad(net, img, label, DEVICE, stdev_spread = 0, num=1)
            #print(grad_map.shape)
            clip_and_save_single_img(grad_map, i * batch_img.size(0) + j, save_dir=save_path)
            #print(grad.shape)
            #simg = (img + mean) * std
            simg = img * std + mean
            #print('rb', simg.max(), simg.min())
            simg = torch.clamp(simg, 0, 1)
            #print('r', simg.max(), simg.min())
            simg = simg.detach().cpu().numpy() * 255.0
            #print(simg.shape)
            #print(simg.shape)
            simg = simg[0]
            simg = np.transpose(simg, (1, 2, 0)).astype(np.uint8)
            #print('r', simg.max(), simg.min())
            #imwrite(os.path.join(save_bench, '{}.png'.format(i * batch_img.size(0) + j)), simg)
            io.imsave(os.path.join(save_bench, '{}.png'.format(i * batch_img.size(0) + j)), simg)
            print(i * batch_img.size(0) + j)

            #grad = imread(os.path.join(save_path, '{}-smooth.png'.format(i * batch_img.size(0) + j)))
            grad = io.imread(os.path.join(save_path, '{}-smooth.png'.format(i * batch_img.size(0) + j)),
                             as_gray = False)
            # if gray
            # grad = grad[:, :, np.newaxis]
            # grad = np.repeat(grad, 3, axis = 2)

            gray_grad = np.mean(grad, axis = -1, keepdims = True)
            gray_grad = gray_grad.astype(np.uint8)
            gray_grad = np.repeat(gray_grad, 3, axis = 2)
            pair_img = np.concatenate((gray_grad, grad, simg), axis=1)
            #imwrite(os.path.join(save_path, '{}-pair.png'.format(i * batch_img.size(0) + j)), pair_img)
            io.imsave(os.path.join(save_path, '{}-pair.png'.format(i * batch_img.size(0) + j)), pair_img)
            labels.append(batch_label.numpy())
    labels = np.array(labels)
    np.savetxt(os.path.join(save_bench, 'label.txt'), labels.reshape(-1))

    #MakeVisual(save_bench, save_path)

def l1_for_without_smooth(net, dl, DEVICE):
    net.eval()
    net.to(DEVICE)
    #criterion = nn.CrossEntropyLoss().to(DEVICE)
    l1s = []
    for i, (batch_img, batch_label) in enumerate(dl):
        #if i> 5:
        #    break
        for j in range(int(batch_img.size(0))):
            img = batch_img[j]
            label = batch_label[j]
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            #print(img.size())
            grad_map = GetSmoothGrad(net, img, label, DEVICE, stdev_spread = 0.05, num=32)
            #print(grad_maps.size(), batch_img.size())
            l1s.append(torch.norm(grad_map, 1).item())
    l1s = np.array(l1s)
    print("Min: {:.4f} -- Max: {:.2f} -- Mean:{:.2f}".format(l1s.min(), l1s.max(), l1s.mean()))


def l1_for_with_smooth(net, dl, DEVICE):
    net.eval()
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    l1s = []
    for i, (batch_img, batch_label) in enumerate(dl):
        batch_img = batch_img.to(DEVICE)
        batch_label = batch_label.to(DEVICE)
        batch_img.requires_grad = True
        pred = net(batch_img)
        loss = criterion(pred, batch_label)
        grad_maps = torch.autograd.grad(loss, batch_img, create_graph=True, only_inputs=False)[0]
        #print(grad_maps.size(), batch_img.size())
        l1s.append(torch.norm(grad_maps, 1).item())
    l1s = np.array(l1s)
    print("Min: {:.2f} -- Max: {:.2f} -- Mean:{:.2f}".format(l1s.min(), l1s.max(), l1s.mean()))


def MakeVisual(data_dir = './benchmark/CIFAR', result_dir = './SmoothRes/CIFAR/'):
    save_p = result_dir.split('/')[:-1]
    save_p = os.path.join(*save_p)
    print(save_p)
    net_name = result_dir.split('/')[-1]
    labels = np.loadtxt(os.path.join(data_dir, 'label.txt'))

    imgs = get_a_set(labels, result_dir, data_dir, times = 3)
    print(os.path.join(save_p, '{}.png'.format(net_name)))
    imwrite(os.path.join(save_p, '{}.png'.format(net_name)), imgs)

def test_model(net, dl):
    accs = []
    net.eval()
    for i, (batch_img, batch_label) in enumerate(dl):
        batch_img = batch_img.to(DEVICE)
        batch_label = batch_label.to(DEVICE)
        pred = net(batch_img)
        acc = torch_accuracy(pred, batch_label)
        accs.append(acc)
    accs = np.array(accs)
    print('accuracy', accs.mean())


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



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type = str,
                        default='../exps/tradeoff.eps8/checkpoint.pth.tar')
    parser.add_argument('-d', type = int, default=3)

    args = parser.parse_args()


    net_name = args.resume.split('/')[-2]
    print(net_name)
    path = os.path.join('../SmoothRes', net_name)
    if not os.path.exists(path):
        os.mkdir(path)
    net = models.resnet18(pretrained=False)
    net.fc.out_features = 200

    net.load_state_dict(torch.load(args.resume)['state_dict'])
    DEVICE = torch.device('cuda:{}'.format(args.d))

    net.to(DEVICE)
    dl = create_test_dataset(32)

    #xz_test(dl, 1,net, DEVICE)
    #test_model(net, dl)
    #l1_for_without_smooth(net, dl, DEVICE)
    #l1_for_with_smooth(net, dl, DEVICE)
    get_result(net, dl, DEVICE, net_name)


