from dataloader import DataLoader
#import skimage.io as io
from cv2  import imwrite, imread
import os
import numpy as np
from collections import namedtuple
import  torch
DATA = namedtuple('Data', field_names=['X', 'Y'])


def gen_benchmark(dl, batch_num = 5, p = './benchmark/MNIST'):
    count = 0
    labels = []
    nb = 0
    for batch_img, batch_label in dl:
        nb = nb + 1
        #print(batch_img.type(), torch.max(batch_img).item(),torch.min(batch_img).item())
        batch_img = batch_img.numpy()
        print(batch_img.min())
        for img in batch_img:
            #print(type(img))
            count = count + 1
            imwrite(os.path.join(p, '{}.png'.format(count)), img[0])

        labels.append(batch_label.numpy())
        if nb >= batch_num:
            break

    labels = np.array(labels)
    labels = labels.reshape(-1)

    np.savetxt(os.path.join(p, 'label.txt'), labels)

def read_data_MNIST(p = './benchmark/Fashion_MNIST'):
    labels = np.loadtxt(os.path.join(p, 'label.txt'))

    imgs = []
    for i in range(1, 161):
        img_path = os.path.join(p, '{}.png'.format(i))
        img = imread(img_path)
        #print(img.shape)
        img = img[:,:,0]
        imgs.append(img)
    imgs = np.array(imgs)
    print('X of shape {} --- Y of shape {}'.format(imgs.shape, labels.shape))
    print('Data from {}'.format(p))
    data = DATA(imgs, labels)
    return data


if __name__ == '__main__':
    #read_data_MNIST()
    DL = DataLoader("Fashion_MNIST", 32, 28)
    _, dl, _, _ = DL.Fashion_MNIST()
    gen_benchmark(dl, p = './benchmark/Fashion_MNIST')