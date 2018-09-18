# Unsupervised generation and classification of image using GANs
D1254 Mengnan Wang
### Requirements
- python==2.7
- numpy==1.14.3
- Keras==1.2.2
- matplotlib==2.0.0
- tensorflow_gpu==1.4.0
- Theano==1.0.0
- six==1.11.0
- Pillow==5.2.0
- scipy==1.1.0
- scikit_learn==0.19.2
- tensorflow==1.10.1
### Weights and dataset
Please download the pre-trained weights for clustering model and dataset from
[Google Drive](https://drive.google.com/drive/folders/18YN_xTWmGUOT0XZDltu6mjVz66Xt_GFw?usp=sharing) and place the folders in this dictionary

### Run
Run `run_experiment.py` and select experiment index. The tensorboard file, checkpoint file and sampled images will be stored in `results` folder
```
>>> python run_experiment
>>>
===Unsupervised generation and classification of image using GANs===
===D1254 Mengnan Wang===

 Experiment index:
 0 DEC + GAN on MNIST
 1 Modified CatGAN on FMNIST
 2 Modified CatGAN + GAN on FMNIST
 3 IMSAT + GAN on SVHN
 4 LearningByAssociation + GAN on SVHN
 5 LearningByAssociation + GAN on CIFAR-10
 6 LearningByAssociation + GAN + VAEGAN on SVHN

(To speed up the implemention, I use a cnn model to
approximate IMSAT and LearningByAssociation instead of building
the computational graph for each model in Tensorflow. The cnn
model is trained with the labels generated by IMSAT and LearningByAssociation.
So it is still an unsupervised/semi-supervised model.)

Please input experiment index...3

```
### Results
- ##### DEC + GAN on MNIST
Experiment 1
![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/mnist-092.jpeg =250x)
Experiment 2
![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/mnist-069.jpeg)
 - ##### Modified CatGAN on FMNIST
 ![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/acc_boost.png)
- ##### Modified CatGAN + GAN on FMNIST
 ![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/fmnist-073.jpg)
- ##### IMSAT + GAN on SVHN
 ![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/svhn-050.jpg)
 - ##### LearningByAssociation + GAN on SVHN
  ![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/svhn-090.jpg)
 - ##### LearningByAssociation + GAN on CIFAR-10
  ![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/cifar-gen.jpg)
- ##### LearningByAssociation + GAN + VAEGAN on SVHN
 ![](https://github.com/wangmn93/Unsupervised-generation-and-classification-of-image-using-GANs/blob/master/experiments_results/vaegan-gen-3.jpg)
