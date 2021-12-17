# UNIST CSE472 Final Project

**Image Classification without Deep Neural Network**

## Abstract

The image classification problem is a problem in which the attributes of an image are classified by a computer, 
and has long been a challenging problem in computer vision area. 
Recently, models that solve image attribute problems based on deep neural networks have shown good results. 
So, what is the way not to use deep neural networks? 
This project aims to find out how to solve the image classification problem without using artificial neural networks and compare results between them.

# Requirements

- numpy, scipy, scikit-learn
- jupyternotebook
- tqdm
- osqp
- cupy (optional)

# Download Dataset

## CIFAR 10

```
mkdir data
cd data
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz .
tar -xvf cifar-10.python.tar.gz
```

## MNIST

[DeepAI Downlaod Homepage](https://deepai.org/dataset/mnist)

Download `mnist.zip` at DeepAI homepage and put in `./data/mnist` directory.
Unzip file as below command.

```
gzip -d .\train-images-idx3-ubyte.gz
gzip -d .\train-labels-idx1-ubyte.gz
gzip -d .\t10k-images-idx3-ubyte.gz
gzip -d .\t10k-labels-idx1-ubyte.gz
```

# Execute Project

Run with `main.ipynb` with jupyter notebook.

# Reference

## Dataset

- [Deep AI](https://deepai.org/dataset/mnist)
- [Parse MNIST](https://stackoverflow.com/questions/39969045/parsing-yann-lecuns-mnist-idx-file-format)

## Model

- [PRML](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) Ch 7.

## Library

- [cupy](https://cupy.dev/)
- [osqp](https://osqp.org/)