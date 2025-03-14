import numpy as np
import scipy.stats as stats
import torch
import copy
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import argparse


def main(args): 

    os.makedirs(os.path.join("experiments", args.path), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
        # build transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        ]) 

    # choose the training and test datasets
    train_data = datasets.MNIST(args.datasets_path, train = True,
                                download = True, transform = transform)
    test_data = datasets.MNIST(args.datasets_path, train=False,
                                download = True, transform = transform)

    train_size = len(train_data)
    test_size = len(test_data)

    # build the data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = args.batch_size, shuffle = False)

    # specify the image classes
    classes = [f"{i}" for i in range(10)]
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Deep VAE Model 
                    Example of usage:
                    python main.py --path base --agent fc --memory_size 1000 --explore --lr 0.1 --epoch 300 --optimizer SGD --epoch_eval 10 --freq 50""")

    # General parameters
    parser.add_argument('--path', type=str, default='base', help='Path to the folder where to model and plots')
    parser.add_argument('--datasets_path', type=str, default='/Users/theofocsa/Desktop/Deep Learning 3/MNIST', help='Path to the dataset MNIST')
    parser.add_argument('--batch_size', type=int, default=128, help='Size of the batch')
    parser.add_argument('')

    args = parser.parse_args()



    main(args)