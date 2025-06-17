from torch.utils import data
from sklearn.datasets import load_digits
from torch import tensor
import torchvision.datasets as datasets
from pynndescent import NNDescent
import os
import joblib
import torch
import numpy as np
from PIL import Image
import scanpy as sc
import scipy
from sklearn.decomposition import PCA
import sklearn
import pandas as pd


from dataloader.data_sourse import DigitsDataset






class MnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:60000]).astype(np.float32) / 255).reshape((60000, -1))
        label = np.array(D.targets[:60000]).reshape((-1))
        return data, label

class MiniMnistDataset(DigitsDataset):
    def load_data(self, data_path, train=True):
        D = datasets.MNIST(root=data_path, train=True, download=True, transform=None)

        data = (np.array(D.data[:30000]).astype(np.float32) / 255).reshape((30000, -1))
        label = np.array(D.targets[:30000]).reshape((-1))
        return data, label


