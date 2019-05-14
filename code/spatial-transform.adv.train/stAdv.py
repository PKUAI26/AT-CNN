import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import torch_accuracy

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
    def forward(self, x):
        '''

        :param x: should be shape of [N, C, H, W, 2]
        :return:
        '''
        batch_size = x.size(0)
        loss = torch.sum(torch.sqrt(1e-7+ torch.sum((x[:,:,1:,:] - x[:, :, :-1, :]) ** 2, dim = -1))) + \
               torch.sum(torch.sqrt(1e-7+ torch.sum((x[:,1:,:,:] - x[:, :-1, :, :]) ** 2, dim = -1)))

        loss = loss /  batch_size

        return loss
class SpatialTransformedAttack(nn.Module):

    #def __init__(self, size = 223, pai = 1, sigma = 0.02, nb_iters = 10):
    def __init__(self, criterion, size=223, pai=1, sigma=0.5, nb_iters=5):
        super(SpatialTransformedAttack, self).__init__()

        self.criterion = criterion
        self.middle = (size )/ 2
        self.size = size
        self.pai = pai
        self.sigma = sigma
        self.nb_iters = nb_iters
        self.TV = TVLoss()
        x = torch.range(0, size)
        y = torch.range(0, size)
        #print(y.size())
        self.x_grid, self.y_grid = torch.meshgrid(x, y)

        #print(self.x_grid.size())
        self.x_grid = (self.x_grid.unsqueeze(0).unsqueeze(-1) - self.middle) / self.middle
        self.y_grid = (self.y_grid.unsqueeze(0).unsqueeze(-1) - self.middle) / self.middle

    def attack(self, net, input, label, *args, **kwargs):

        net.eval()
        input = input.detach()

        flow = torch.zeros(input.size(0), input.size(2), input.size(3), 2)
        flow = flow.to(input.get_device())
        flow.requires_grad = True

        x_grid = self.x_grid.to(flow.get_device())
        y_grid = self.y_grid.to(flow.get_device())
        grid = torch.cat((x_grid, y_grid), dim = -1)


        optimizer = torch.optim.LBFGS([flow], lr = self.sigma, max_iter=10)
        def closure():
            optimizer.zero_grad()
            locations = grid + flow
            #print(flow.max())
            adv_inp = F.grid_sample(input, locations)

            loss_cls = -1.0 * self.criterion(net(adv_inp), label)
            loss_f = self.TV(flow)

            loss = loss_cls + self.pai * loss_f

            loss.backward()
            #print(flow.grad)
            return loss

        for i in range(self.nb_iters):
            optimizer.step(closure)

        locations = grid + flow

        adv_inp = F.grid_sample(input, locations)

        '''
        for i in range(self.nb_iters):

            #print(grid, flow)
            locations = grid + flow

            adv_inp = F.grid_sample(input, locations)

            loss_cls = -1.0 * self.criterion(net(adv_inp), label)
            loss_f = self.TV(flow)

            loss = loss_cls + self.pai * loss_f

            #loss = loss_cls
            print(loss.item() , loss_f.item(), loss_cls.item())
            loss.backward()

            grad_sign = torch.sign(flow.grad)
            #print(flow.grad)

            flow = flow - grad_sign * self.sigma

            flow = flow.detach()
            flow.requires_grad = True
        locations = grid + flow
        return F.grid_sample(input, locations)
        '''

        return adv_inp

def test():
    import torchvision.models as models
    DEVICE = torch.device('cuda:{}'.format(0))

    net = models.resnet18(pretrained=False)
    net.fc = nn.Linear(512, 257)
    net.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    x =  torch.ones(5, 3, 224, 224).to(DEVICE)
    labels = torch.ones(5).type(torch.long).to(DEVICE)

    attack = SpatialTransfromedAttack()


    adv_inp = attack(x, labels, net, criterion)

    print(adv_inp.size())

if __name__ == '__main__':
    test()
