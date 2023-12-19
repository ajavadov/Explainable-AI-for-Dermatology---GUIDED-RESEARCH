import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import cv2

from scipy import stats
import skimage
import imageio
from PIL import Image

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

class ISIC2019BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_path,
                 transform=None,
                 ds_type=None,
                 md_choice='all') -> None:
        super().__init__()

        self.diagnostic_label = ['NV','BCC','MEL','AK','SCC','BKL','VASC','DF']

        self.md_lbl_cat_dict = {
            'anatom_site_general' : ['anterior torso',
                                    'lower extremity',
                                    'head/neck',
                                    'upper extremity',
                                    'posterior torso',
                                    'palms/soles',
                                    'oral/genital',
                                    'lateral torso'],
            'sex' : ['male', 'female'],
        }

        self.md_lbl_num_dict = {
            'age_approx' : [],
        }

        self.root_path = root_path

        self.transform = transform

        assert ds_type in ['train','val','test'], "Please provide the type of dataset"

        if ds_type == 'train':
            self.dataframe = pd.read_csv(os.path.join(root_path,'train_data.csv'))
            self.meta = pd.read_csv(os.path.join(root_path,'train_meta.csv'))[['anatom_site_general','sex','age_approx']]
        elif ds_type == 'test' or ds_type == 'val':
            self.dataframe = pd.read_csv(os.path.join(root_path,'test_data.csv'))
            self.meta = pd.read_csv(os.path.join(root_path,'test_meta.csv'))[['anatom_site_general','sex','age_approx']]

        self.dataframe = self.dataframe[['image'] + self.diagnostic_label]

        if md_choice == 'all':
            self.md_lbl_list = list(self.md_lbl_cat_dict.keys()) + list(self.md_lbl_num_dict.keys())
        elif type(md_choice) is list:
            self.md_lbl_list = md_choice
        else:
            md_corr_sort, md_nr_lbl = md_choice.split(':')

            assert md_corr_sort in ['worst', 'best'], 'Please provide the correct direction of sorting'

            md_nr_lbl = int(md_nr_lbl)
            correlation_matrix = pd.read_csv(os.path.join(root_path,'corr_mat.csv'), index_col=0)
            correlation_vector = correlation_matrix['diagnostic'].drop(['diagnostic', 'biopsed', 'patient_id', 'lesion_id', 'img_id'], axis=0) \
                                                                        .abs() \
                                                                        .sort_values(ascending=True if md_corr_sort == 'worst' else False)
            self.md_lbl_list = list(correlation_vector.index)[:md_nr_lbl]

        # self.targets = self.dataframe.iloc[:,2:-1].idxmax(axis='columns')
        self.targets = self.dataframe.iloc[:,1:].idxmax(axis='columns')

    def __len__(self):
        return len(self.dataframe)

    def to_categorical(self, y, num_classes=None, soft_labels=True, dtype='float32'):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((n, num_classes), dtype=dtype)
        if soft_labels:
            categorical += 0.1
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (num_classes,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
    
    def encode_meta_label(self, img_info, meta_data_names):

        meta_vector = []

        for key in meta_data_names:
            if key in self.md_lbl_num_dict.keys():
                continue
            value = self.md_lbl_cat_dict[key]
            meta_vector.append(self.to_categorical(value.index(str(img_info[key])), len(value)))

        for key in meta_data_names:
            if key in self.md_lbl_cat_dict.keys():
                continue
            value = self.md_lbl_num_dict[key]
            meta_vector.append(np.float32(img_info[key] / 100.0))
            
        meta_vector = np.hstack(meta_vector)

        return meta_vector

    def __getitem__(self, idx):

        row_data = self.dataframe.iloc[idx]
        lbl = self.targets[idx]
        row_meta = self.meta.iloc[idx]

        img_path = os.path.join(self.root_path, 'ISIC_2019_Training_Input', row_data['image']+'.jpg')

        try:
            img = skimage.io.imread(img_path, plugin='pil')[:,:,:3]
        except FileNotFoundError as err:
            print("We have an error")
            print(err)
            raise

        img = cv2.resize(img, (224,224))

        img_no_norm = torch.from_numpy(np.transpose(np.array(img), (2,0,1)))
        # img_no_norm = torch.from_numpy(np.transpose(np.array(img), (1,2,0)))

        if self.transform:
            img = self.transform(image=img)['image']

        img = torch.from_numpy(np.transpose(img,(2,0,1)).astype('float32'))
        # img = torch.from_numpy(np.transpose(img,(1,2,0)).astype('float32'))


        label = np.array(self.diagnostic_label.index(lbl)).astype(np.int64)
        
        meta_data = self.encode_meta_label(row_meta, self.md_lbl_list)

        meta_data = torch.from_numpy(meta_data)

        return img_no_norm, img, meta_data, label

class ISIC2019Dataset(pl.LightningDataModule):
    def __init__(self,
                 data_dir: str = None,
                 batch_size: int = 16,
                 sampling_rate: int = -1,
                 normalize_weights: bool = True,
                 md_choice='all'):
        super().__init__()

        if data_dir is None:
            raise Exception("Please provide a path to the dataset")
        
        self.data_dir = data_dir
        
        self.batch_size = batch_size

        self.normalize_weights = normalize_weights

        self.sampling_rate = sampling_rate

        norm_mean = [0.485, 0.456, 0.406]
        norm_std = [0.229, 0.224, 0.225]

        self.transform_train = Compose(
            [
                VerticalFlip(p=0.5),
                HorizontalFlip(p=0.5),
                ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
                RandomRotate90(p=0.5),
                RandomBrightnessContrast(p=0.5),
                # CenterCrop(),
                # Affine(),
                # GridDistortion(),
                # OpticalDistortion(),
                # RandomGamma(p=0.5),
                # Sharpen(p=0.5),
                # RandomShadow(),
                # ISONoise(),
                # GaussianBlur()
                # GlassNoise()
                # ColorJitter(),
                Normalize(mean=norm_mean,std=norm_std,always_apply=True),
            ])
        
        self.transform_test = Compose(
            [
                Normalize(mean=norm_mean,std=norm_std),
            ])

        self.train_data = ISIC2019BaseDataset(data_dir, transform=self.transform_train, ds_type='train', md_choice=md_choice)
        self.test_data = ISIC2019BaseDataset(data_dir, transform=self.transform_test, ds_type='test', md_choice=md_choice)

        self.label_size = len(self.train_data.diagnostic_label)
        self.num_classes = len(self.train_data.diagnostic_label)

        self.md_lbl_list = self.train_data.md_lbl_list
        concatenated_dict = {**self.train_data.md_lbl_cat_dict, **self.train_data.md_lbl_num_dict}
        self.md_lbl_sizes = [len(concatenated_dict[key]) for key in self.train_data.md_lbl_list]
        self.md_lbl_sizes[-1] += 1

        print(f"{md_choice}: {self.md_lbl_list}")


    def prepare_data(self):
        pass

    def setup(self, stage=None):
        labels, nrs = np.unique(self.train_data.targets, return_counts=True)
        if stage == 'fit' or stage is None:
                labels, nrs = np.unique(self.train_data.targets, return_counts=True)

                order = [self.train_data.diagnostic_label.index(lbl) for lbl in labels]
                labels = [labels[i] for i in order]
                nrs = [nrs[i] for i in order]

                self.class_weights = 1. / np.array(nrs).astype(np.float32)

                if self.sampling_rate >= 0:
                    train_samples_weight = torch.from_numpy(np.array([self.class_weights[labels.index(self.train_data.targets[i])] for i in range(len(self.train_data))]))
                    self.train_sampler = torch.utils.data.WeightedRandomSampler(train_samples_weight, int(self.sampling_rate*len(self.train_data)), replacement=True)
                
                if self.normalize_weights:
                    self.class_weights /= np.sum(self.class_weights)
            
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data,
                                           sampler=self.train_sampler if self.sampling_rate >= 0 else None,
                                           batch_size=self.batch_size,
                                           num_workers=self.batch_size,
                                           drop_last=True,
                                           shuffle=False if self.sampling_rate >= 0 else True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.batch_size,
                                           drop_last=False,
                                           shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.batch_size,
                                           shuffle=False,
                                           drop_last=False)