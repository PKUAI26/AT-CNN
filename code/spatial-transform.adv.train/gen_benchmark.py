#import skimage.io as io
from skimage.io import imsave, imread
import os
import numpy as np
from collections import namedtuple
import  torch
from dataset import create_test_dataset, create_style_test_dataset, create_saturation_test_dataset, create_patch_test_dataset, \
        create_lowpass_test_dataset, create_highpass_test_dataset
DATA = namedtuple('Data', field_names=['X', 'Y'])


def gen_benchmark(dl, batch_num = 10, p = '../../data/benchmark/val'):
    count = 0
    labels = []
    nb = 0
    _mean = torch.tensor(np.array([0.485, 0.456, 0.406]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    _var = torch.tensor(np.array([0.229, 0.224, 0.225]).astype(np.float32)[np.newaxis, :, np.newaxis, np.newaxis])
    if not os.path.exists(p):
        os.mkdir(p)

    for batch_img, batch_label in dl:
        nb = nb + 1
        batch_img = batch_img * _var + _mean
        #print(batch_img.type(), torch.max(batch_img).item(),torch.min(batch_img).item())
        batch_img = batch_img.numpy()
        print(batch_img.min())
        for img in batch_img:
            #print(type(img))
            #print(img.shape)
            count = count + 1
            img = (img * 255).astype(np.uint8)
            img = img.transpose(1,2,0)
            imsave(os.path.join(p, '{}.png'.format(count)), img)

        labels.append(batch_label.numpy())
        if nb >= batch_num:
            break

    labels = np.array(labels)
    labels = labels.reshape(-1)

    np.savetxt(os.path.join(p, 'label.txt'), labels)




if __name__ == '__main__':

    root = '../../data/benchmark/'
    '''
    names = ['val', 'style', 'sat1024', 'sat1', 'sat64', 'sat16', 'sat8', 'sat4', 'p2', 'p4', 'p8']
    dl_val = create_test_dataset(batch_size = 64)
    dl_style = create_style_test_dataset(batch_size = 64)
    dl_sat1024 = create_saturation_test_dataset(batch_size=64, saturation_level = 1024)
    dl_sat02 = create_saturation_test_dataset(batch_size=64, saturation_level=1)
    dl_sat64 = create_saturation_test_dataset(batch_size=64, saturation_level=64)
    dl_sat16 = create_saturation_test_dataset(batch_size=64, saturation_level=16)
    dl_sat8 = create_saturation_test_dataset(batch_size=64, saturation_level=8)
    dl_sat4 = create_saturation_test_dataset(batch_size=64, saturation_level=4)
    dl_p2 = create_patch_test_dataset(64, k = 2)
    dl_p4 = create_patch_test_dataset(64, k = 4)
    dl_p8 = create_patch_test_dataset(64, k = 8)

    #dls = [dl_val, dl_style, dl_sat1024, dl_sat02, dl_sat64, dl_sat16, dl_sat8, dl_sat4, dl_p2, dl_p4, dl_p8]
    '''
    names = ['high-142','high-144','high-146','high-148',]
    #low120 = create_lowpass_test_dataset(64, r = 120)
    #low44 = create_lowpass_test_dataset(64, r =44)
    high42 = create_highpass_test_dataset(64, r = 142)
    high44 = create_highpass_test_dataset(64, r = 144)
    high46 = create_highpass_test_dataset(64, r = 146)
    high48 = create_highpass_test_dataset(64, r = 148)
    dls = [high42, high44, high46, high48]

    for name, dl in zip(names, dls):
        dir = os.path.join(root, name)
        print('making {}'.format(dir))
        gen_benchmark(dl, p = dir)
