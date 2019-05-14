import numpy as np
import os
import random
import shutil
def partition_one_class(class_name, ratio):
    class_dir = os.path.join('256_ObjectCategories', class_name)
    img_names = os.listdir(class_dir)

    target_dir = os.path.join('./val', class_name)
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    num_val = int(len(img_names) * ratio)

    val_names = random.sample(img_names, num_val)
    for val_name in val_names:
        source_name = os.path.join(class_dir, val_name)
        target_name = os.path.join(target_dir, val_name)
        shutil.move(source_name, target_name)
        print('Moving {} to {}'.format(source_name, target_name))


def PartitionValidation(ratio = 0.2):
    all_class = os.listdir('./256_ObjectCategories')

    for class_name in all_class:
        print('Working on {}'.format(class_name))
        partition_one_class(class_name, ratio)

    print('Done')


if __name__ == '__main__':
    PartitionValidation()
