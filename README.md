# Variational Auto Encoder(VAE) and Latent Interpolation Demo

This project implements VAE from scratch, and trains the model with CIFAR 10 images to demonstrate the latent interpolation from learned latent space. 

## Overview
The model was built from scratch with PyTorch. Using the assumed latent distribution as Guassian distribution. The goal of this project is to:
* build the actual model from scratch
* train the model with CIFAR 10 image dataset
* learn the continuous latent space the images are encded into
* sample  continuous latent vectors/ interpolate them/ and decode them back to image space


### How to Run
```
pip install -r requirements.txt
git clone https://github.com/markna627/vae.git
cd vae
python3 train.py
```

Available Arguments:
```
--layers   (int) the number of encoder/decoder stacks
--heads    (int) the number of heads in the multi-head attention
--epochs   (int) the number of epochs
```





### Example:

#### Generation - Decoder Performance
![Training/Validatation Loss](/assets/generated.png)

#### Interpolation between Samples
![Training/Validatation Loss](/assets/interpolation.png)

### Notes 
Colab demo is available:
[Here](https://colab.research.google.com/drive/1BywK8P9n4dc02KBK_xsI7k1_YT2fkqUC?usp=sharing)


