import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cv2

from scipy import stats
import skimage
import scipy
import imageio
from PIL import Image
from skimage import io
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

import pandas as pd
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


import torchvision
import torchvision.transforms as transforms

from pytorch_metric_learning.samplers import MPerClassSampler
import torchsample

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    ShiftScaleRotate,
    RandomBrightnessContrast,
    Normalize
)

aug = Compose(
    [
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.5),
        #RandomContrast(p=0.5),
        #RandomBrightness(p=0.5),
        #RandomGamma(p=0.5)

    ],
    p=0.5)


from .seven_point_dataset_utils import *
# from .seven_point_utils import encode_label, encode_meta_label

class SevenPointBaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path,
                 transform=None,
                 type=None,
                 use_metadata=0) -> None:
        super().__init__()

        self.root_path = root_path

        self.transform = transform 

        self.use_metadata = use_metadata

        # self.dataframe = pd.read_csv(os.path.join(root_path,'meta/meta.csv'))
        self.dataframe = pd.read_csv(os.path.join(root_path,'meta/meta_mel_or_not.csv'))
        self.train_indices = pd.read_csv(os.path.join(root_path,'meta/train_indexes.csv'))
        self.val_indices = pd.read_csv(os.path.join(root_path,'meta/valid_indexes.csv'))
        self.test_indices = pd.read_csv(os.path.join(root_path,'meta/test_indexes.csv'))
        # self.test_indices = pd.read_csv(os.path.join(root_path,'meta/test_indexes_2.csv'))
        #self.test_indices = pd.read_csv(os.path.join(root_path,'meta/test_indexes_3.csv'))


        #new version of test indices composed of each class member (e.g. 2 per each) + if i find doctor marked images

        #(next step) look at papers to see how people treat gradcam output
        print("AA",self.test_indices)

        assert type in ['train','val','test'], "Please provide the type of dataset"

        if type == 'train':
            self.dataframe = self.dataframe.iloc[self.train_indices['indexes'].values]
        if type == 'val':
            self.dataframe = self.dataframe.iloc[self.val_indices['indexes'].values]
        if type == 'test':
            self.dataframe = self.dataframe.iloc[self.test_indices['indexes'].values]

        self.targets = []

        for i in range(len(self.dataframe)):
            print(self.dataframe)
            print(self.dataframe.iloc[i])
            img_info = self.dataframe.iloc[i]

            diagnosis_label = img_info['diagnosis']
            for index, label in enumerate(label_list_2):
                if diagnosis_label in label:
                    self.targets.append(index)
                    break
                else:
                    continue

        self.targets = np.array(self.targets)
        print(self.targets)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):

        row = self.dataframe.iloc[idx]

        img_derm_path = os.path.join(self.root_path, 'images', row['derm'])

        try:
            img_derm = skimage.io.imread(img_derm_path)
        except FileNotFoundError as err:
            print("We have an error")
            print(err)
            raise

        img_derm = cv2.resize(img_derm, (224,224))

        img_derm_original = np.array(img_derm).astype('float32')
        img_derm_original /= img_derm_original.max()

        if self.transform:
            img_derm = self.transform(image=img_derm)['image']

        img_derm = torch.from_numpy(np.transpose(img_derm,(2,0,1)).astype('float32'))# / 255.0
        img_derm_original = torch.from_numpy(np.transpose(img_derm_original,(2,0,1)).astype('float32'))# / 255.0

        labels = encode_label(row)

        meta_data = encode_meta_label(row, use_metadata=self.use_metadata)

        meta_data = torch.from_numpy(meta_data)

        return img_derm_original, img_derm, meta_data, labels

class SevenPointDataset(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = None,
                 batch_size: int = 16,
                 normalize: bool = True,
                 print_log: bool = False,
                 normalize_weights: bool = True,
                 use_metadata='all'
                 ):
        super().__init__()

        if data_dir is None:
            raise Exception("Please provide a path to the dataset")
        
        self.data_dir = data_dir
        
        self.batch_size = batch_size

        self.normalize_weights = normalize_weights


        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        if normalize:
            self.transform_train = Compose(
                [
                    VerticalFlip(p=0.5),
                    HorizontalFlip(p=0.5),
                    ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
                    RandomRotate90(p=0.5),
                    RandomBrightnessContrast(p=0.5),
                    Normalize(mean=norm_mean,std=norm_std,always_apply=True),
                ])
        else:
            self.transform_train = Compose(
                [
                    VerticalFlip(p=0.5),
                    HorizontalFlip(p=0.5),
                    ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
                    RandomRotate90(p=0.5),
                    RandomBrightnessContrast(p=0.5)
                ])
        
        self.transform_test = Compose(
            [
                Normalize(mean=norm_mean,std=norm_std),
            ])

        self.train_data = SevenPointBaseDataset(data_dir, transform=self.transform_train, type='train', use_metadata=use_metadata)
        self.val_data = SevenPointBaseDataset(data_dir, transform=self.transform_test, type='val', use_metadata=use_metadata)
        self.test_data = SevenPointBaseDataset(data_dir, transform=self.transform_test, type='test', use_metadata=use_metadata)

        self.num_classes = len(np.unique(self.train_data.targets))
        print("Number of Classes:",self.num_classes)
        self.meta_label_dict = meta_label_categorical_no_nan

        self.meta_data_sizes = meta_data_sizes
        self.meta_data_labels = meta_data_labels

        if use_metadata == 'all':
            use_metadata = len(self.meta_data_sizes)
        else:
            self.meta_data_sizes = self.meta_data_sizes[:use_metadata]
            self.meta_data_labels = self.meta_data_labels[:use_metadata]

        

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        if stage == 'fit' or stage is None:            
            self.class_weights = 1. / np.unique(self.train_data.targets, return_counts=True)[1].astype(np.float32)
            train_samples_weight = torch.from_numpy(np.array([self.class_weights[self.train_data.targets[i]] for i in range(len(self.train_data))]))
            self.train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight, 3*len(self.train_data), replacement=True)
            if self.normalize_weights:
                self.class_weights /= np.sum(self.class_weights)
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           batch_size=self.batch_size,
                                           num_workers=8,
                                           drop_last=False,
                                           shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data,
                                           batch_size=self.batch_size,
                                           num_workers=8,
                                           drop_last=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           num_workers=8,
                                           shuffle=False,
                                           drop_last=True)


class SevenPointSubset(torch.utils.data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset

    From : https://stackoverflow.com/questions/51782021/how-to-use-different-data-augmentation-for-subsets-in-pytorch
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img_clin, img_derm, meta_data, labels = self.dataset[self.indices[idx]]
        return img_clin, img_derm, meta_data, labels

    def __len__(self):
        return len(self.indices)
