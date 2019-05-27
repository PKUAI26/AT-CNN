# Interpreting Adversarially Trained Convolutional Neural Networks

Code for our [paper](https://arxiv.org/abs/1905.09797): Interpreting Adversarially Trained Convolutional Neural Networks" by [Tianyuan Zhang](http://tianyuanzhang.com), [Zhanxing Zhu](https://sites.google.com/view/zhanxingzhu).



# How to run
If you want to run code on your own, I only provide codes on Caltech-256.
We do not upload the dataset and trainned networks due to the fact that they are storage consuming.
Training(Adversarial Training) scripts in this repo is not well writen, I suugest you to use your own scripts, or scripts provided in [this repo](https://github.com/a1600012888/YOPO-You-Only-Propagate-Once)



## Caltech-256

### The four most important python files for Caltech-256

They are all in  `/code/baseline/` :

 `main.py`, `attack.py`, `utils.py` `dataset.py`

1. `main.py`  trains CNNs.  It contains sufficient comments to understand how to customize your trainings.
2. `attack.py` implements PGD and FGSM attackers.
3. `utils.py` implements **SmoothGrad** 
4. `dataset.py` implements **Saturation** and **Patch-shuffle** operation. For style transfer, you have to use code provided in https://github.com/rgeirhos/Stylized-ImageNet/tree/master/code
5. `stAdv.py` in code/spatial-transform.adv.train/ implements Spatially transformed attack.



### How to run

1. Dowload the data from  http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html to `/code/Caltech256/data`  
2. `cd data`   and run `Partition.py` to generate training set and test set.
3. `cd code/baseline` and run `main.py` to train standard CNNs
4.  `cd code/pgd.inf.eps8` and run `main.py` to adversarially train CNNs against a $l_{\inf}$ -norm bounded PGD attacker. code in `/code/Caltech256/code/pgd.l2.eps8`  are for $l_2$ -norm bounded adversarial training .
5.  `gen_visual.py` and ` utils.py` in  `/code/Caltech256/code/baseline/` contains code to generate salience maps.



## TinyImageNet & CIFAR-10

Code for TinyImageNet, and CIAFR-10 are similar to that for `Caltech-256`,  important python files such as `main.py`, `utils.py`, `attack.py` has the same name and functionallities

You still have to download the images from https://tiny-imagenet.herokuapp.com and put them to `/code/TinyImageNet/data`



   

