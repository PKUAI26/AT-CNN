## README

If you want to run code on your own, I stongly recomend you to run code on Caltech-256, because code for experiements on Caltech-256 are the most well structured one. 

We do not upload the dataset and trainned networks due to the fact that they storage consuming.



## Caltech-256

### The four most important python files for Caltech-256

They are all in  `/code/Caltech256/code/baseline/` :

 `main.py`, `attack.py`, `utils.py` `dataset.py`

1. `main.py`  trains CNNs.  It contains sufficient comments to understand how to customize your trainings.
2. `attack.py` implements PGD and FGSM attackers.
3. `utils.py` implements **SmoothGrad** 
4. `dataset.py` implements **Saturation** and **Patch-shuffle** operation. For style transfer, you have to use code provided in https://github.com/rgeirhos/Stylized-ImageNet/tree/master/code



### How to run

1. Dowload the data from  http://www.vision.caltech.edu/Image_Datasets/Caltech101/Caltech101.html to `/code/Caltech256/data`  
2. `cd /code/Caltech256/data`   and run `Partition.py` to generate training set and test set.
3. `cd /code/Caltech256/code/baseline` and run `main.py` to train standard CNNs
4.  `cd /code/Caltech256/code/pgd.inf.eps8` and run `main.py` to adversarially train CNNs against a $l_{\inf}$ -norm bounded PGD attacker. code in `/code/Caltech256/code/pgd.l2.eps8`  are for $l_2$ -norm bounded adversarial training .
5.  `gen_visual.py` and ` utils.py` in  `/code/Caltech256/code/baseline/` contains code to generate salience maps.



## TinyImageNet

Code for TinyImageNet are similar to that for `Caltech-256`,  important python files such as `main.py`, `utils.py`, `attack.py` has the same name and functionallities

You still have to download the images from https://tiny-imagenet.herokuapp.com and put them to `/code/TinyImageNet/data`



## CIFAR-10

Codes for CIFAR-10 still has similar structures with that for Caltech256 and TinyImageNet, but due to the fact that code for CIFAR-10 is written log ago, so they are not well strcutured.



`/code/Cifar/code/baseline` contains all the files for CIFAR10

1. `main.py` , `utils.py`, `attack.py` have the same functionality with that for Caltech-256 and TinyImageNet

   

