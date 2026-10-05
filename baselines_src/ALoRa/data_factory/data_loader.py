import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os

class PSMSegLoaderFull(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        # test_data = test_data.iloc[:,:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:  # Assuming overlapping windows with step size of 1
            return self.test.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            index = index * self.step
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            index = index * self.step
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            index = index * self.step
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:  # Overlapping windows with step size of 1
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
class HAISegLoaderFull(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        # test_data = test_data.iloc[:,:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_labels.csv').values[:, :]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    # def __len__(self):
    #     """
    #     Number of images in the object dataset.
    #     """
    #     if self.mode == "train":
    #         return (self.train.shape[0] - self.win_size) // self.step + 1
    #     elif (self.mode == 'val'):
    #         return (self.val.shape[0] - self.win_size) // self.step + 1
    #     elif (self.mode == 'test'):
    #         return (self.test.shape[0] - self.win_size) // self.step + 1
    #     else:
    #         return (self.test.shape[0] - self.win_size) // self.win_size + 1

    # def __getitem__(self, index):
    #     index = index * self.step
    #     if self.mode == "train":
    #         return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    #     elif (self.mode == 'val'):
    #         return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
    #     elif (self.mode == 'test'):
    #         return np.float32(self.test[index:index + self.win_size]), np.float32(
    #             self.test_labels[index:index + self.win_size])
    #     else:
    #         return np.float32(self.test[
    #                           index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
    #             self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:  # Assuming overlapping windows with step size of 1
            return self.test.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            index = index * self.step
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            index = index * self.step
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            index = index * self.step
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:  # Overlapping windows with step size of 1
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])


class SWatSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train_minute.csv')
        data = data.values[:, 1:] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test_minute.csv')

        test_data = test_data.values[:, 1:]
        # test_data = test_data.iloc[:,:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_minute_labels.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)


    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:  # Assuming overlapping windows with step size of 1
            return self.test.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            index = index * self.step
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            index = index * self.step
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            index = index * self.step
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:  # Overlapping windows with step size of 1
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/MSL_train.npy")[0:,:] 
        # data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")[0:,:]
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:  # Assuming overlapping windows with step size of 1
            return self.test.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            index = index * self.step
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            index = index * self.step
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            index = index * self.step
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:  # Overlapping windows with step size of 1
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train",add_noise=True):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        # self.add_noise = add_noise

        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, :] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:  # Assuming overlapping windows with step size of 1
            return self.test.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            index = index * self.step
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            index = index * self.step
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            index = index * self.step
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:  # Overlapping windows with step size of 1
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])



class MSDSSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:  # Assuming overlapping windows with step size of 1
            return self.test.shape[0] - self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            index = index * self.step
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            index = index * self.step
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            index = index * self.step
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:  # Overlapping windows with step size of 1
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])

def get_loader_segment(data_path, batch_size, win_size=100, step=1, mode='train', dataset='SMD'):
    if dataset == 'SMD' : 
        dataset = SMDSegLoader(data_path, win_size, step, mode)
    elif dataset == 'HAI' : 
        dataset = HAISegLoaderFull(data_path, win_size, step, mode)
    elif dataset == 'SWAT':
        dataset = SWatSegLoader(data_path, win_size, step, mode)
    elif dataset == 'MSL':
        dataset = MSLSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'PSM':
        dataset = PSMSegLoaderFull(data_path, win_size, 1, mode)
    elif dataset == 'SMAP':
        dataset = SMAPSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'MSDS':
        dataset = MSDSSegLoader(data_path, win_size, 1, mode)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    shuffle = False
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader


