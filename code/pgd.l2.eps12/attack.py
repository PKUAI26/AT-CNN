import torch
import numpy as np
from utils import torch_accuracy

def clip_eta(eta, norm, eps, is_gpu = True, DEVICE = torch.device('cuda:0')):
    '''
    helper functions to project eta into epsilon norm ball
    :param eta: Perturbation tensor (should be of size(N, C, H, W))
    :param norm: which norm. should be in [1, 2, np.inf]
    :param eps: epsilon, bound of the perturbation
    :return: Projected perturbation
    '''

    assert norm in [1, 2, np.inf], "norm should be in [1, 2, np.inf]"

    with torch.no_grad():
        avoid_zero_div = torch.tensor(1e-12)
        eps = torch.tensor(eps)
        one = torch.tensor(1.0)
        if is_gpu:
            avoid_zero_div = avoid_zero_div.to(DEVICE)
            eps = eps.to(DEVICE)
            one = one.to(DEVICE)
        if norm == np.inf:
            eta = torch.clamp(eta, -eps, eps)
        else:
            normalize = torch.norm(eta.reshape(eta.size(0), -1), p = norm, dim = -1, keepdim = False)
            #print(normalize.size())
            #print(eps.size())
            normalize = torch.max(normalize, avoid_zero_div)
            #normalize = torch.unsqueeze()
            normalize.unsqueeze_(dim = -1)
            normalize.unsqueeze_(dim=-1)
            normalize.unsqueeze_(dim=-1)
            #print(torch.mean(normalize), normalize.size())
            #print(eps)
            #print(normalize.size())
            factor = torch.min(one, eps / normalize)
            #factor = eps / normalize
            #print('eta', eta.size(), factor.size())

            eta = eta * factor

    return eta


def fsgm(net, inp, label, sigma = 1.0):

    #net.cuda()
    net.eval()
    inp.requires_grad = True
    net.zero_grad()
    criterion = torch.nn.CrossEntropyLoss().cuda()

    pred = net(inp)
    loss = criterion(pred, label)

    loss.backward()

    grad_sign = inp.grad.sign()

    img = inp + sigma * grad_sign

    return img


class IPGD(object):

    _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    _var = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    def __init__(self, eps = 6 / 255.0, sigma = 3 / 255.0, nb_iter = 20, norm = np.inf, DEVICE = torch.device('cuda:0')):
        '''
        :param eps: maximum distortion of adversarial examples
        :param sigma: single step size
        :param nb_iter: number of attack iterations
        :param norm: which norm to bound the perturbations
        '''
        self.eps = eps
        self.sigma = sigma
        self.nb_iter = nb_iter
        self.norm = norm
        self.criterion = torch.nn.CrossEntropyLoss().to(DEVICE)
        self.DEVICE = DEVICE
        self._mean = self._mean.to(DEVICE)
        self._var = self._var.to(DEVICE)

    def single_attck(self, net, inp, label, eta, target = None):
        '''
        Given the original image and the perturbation computed so far, computes
        a new perturbation.
        :param net:
        :param inp: original image
        :param label:
        :param eta: perturbation computed so far
        :return: a new perturbation
        '''
        #inp.retain_grad()
        adv_inp = inp + eta
        #adv_inp.retain_grad()
        #print("  eq", adv_inp.requires_grad)
        #adv_inp.requires_grad = True
        net.zero_grad()

        pred = net(adv_inp)
        #loss.backward()
        if target is not None:
            targets = torch.sum(pred[:, target])
            #print(targets.size())
            grad_sign = torch.autograd.grad(targets, adv_in, only_inputs=True, retain_graph = False)[0].sign()

            #print(grad_sign.size(), torch.sum(grad_sign))
            #grad_sign = grad_sign.sign()
        else:
            loss = self.criterion(pred, label)
            #grad_sign = torch.autograd.grad(loss, adv_inp, only_inputs = False)[0].sign()
            grad_sign = torch.autograd.grad(loss, adv_inp,
                                            only_inputs=True, retain_graph = False)[0].sign()
            #grad_sign = torch.autograd.backward(adv_inp, loss)
            #loss.backward()
            ##grad_sign = adv_inp.grads.sign()
            #print(np.sum(grad_sign))
            #print(torch.sum(grad_sign))
        #print(loss.item())

        #print("dd", grad_sign)
        #grad_sign = adv_inp.grad.sign()

        adv_inp = adv_inp + grad_sign * (self.sigma / self._var)
        tmp_adv_inp = adv_inp * self._var +  self._mean

        tmp_inp = inp * self._var + self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1) ## clip into 0-1
        #tmp_adv_inp = (tmp_adv_inp - self._mean) / self._var
        tmp_eta = tmp_adv_inp - tmp_inp
        tmp_eta = clip_eta(tmp_eta, norm=self.norm, eps=self.eps, DEVICE=self.DEVICE)

        eta = tmp_eta/ self._var

        return eta

    def attack(self, net, inp, label, target = None):

        #print(torch.norm(inp[0].reshape(-1), 2), inp[0].size())
        eta = torch.zeros_like(inp)
        eta = eta.to(self.DEVICE)
        net.eval()

        inp.requires_grad = True
        eta.requires_grad = True
        for i in range(self.nb_iter):
            eta = self.single_attck(net, inp, label, eta, target)
            #print(i)

        #print(eta.max())
        adv_inp = inp + eta
        tmp_adv_inp = adv_inp * self._var +  self._mean
        tmp_adv_inp = torch.clamp(tmp_adv_inp, 0, 1)
        adv_inp = (tmp_adv_inp - self._mean) / self._var

        return adv_inp

    def get_batch_accuracy(self, net, inp, label):

        adv_inp = self.attack(net, inp ,label)

        pred = net(adv_inp)

        accuracy = torch_accuracy(pred, label, (1, ))[0].item()


        return accuracy

def test_clip():

    a = torch.rand((10, 3, 28, 28)).cuda()

    epss = [0.1, 0.5, 1]

    norms = [1, 2, np.inf]
    for e, n in zip(epss, norms):
        print(e, n)
        c = clip_eta(a, n, e, True)

        print(c)

def test_IPGD():
    model_path = './exps/exp0/model_best.pth.tar'

    from base_model.cifar_resnet18 import cifar_resnet18
    from dataset import Dataset
    from my_snip.base import EpochDataset

    pgd = IPGD()



    net = cifar_resnet18()
    net.load_state_dict(torch.load(model_path)['state_dict'])

    net.cuda()

    ds_train = Dataset(dataset_name='train')
    ds_train.load()
    ds_train = EpochDataset(ds_train)

    epoch = next(ds_train.epoch_generator())

    count = 0
    for mn_batch in epoch:

        count += 1
        if count > 100:
            break

        data = mn_batch['data']

        label = mn_batch['label']

        #print(data.shape)
        data = torch.tensor(data, dtype=torch.float32).cuda()
        label = torch.tensor(label, dtype=torch.int64).cuda()

        acc = pgd.get_batch_accuracy(net, data, label)

        print('acc: {:.2f}%'.format(acc))

if __name__ == '__main__':
    test_IPGD()
