from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms.functional as F
import torch
import torchio as tio
import nibabel as nib
from utils import *

dict_transform = {'embryo': transforms.Compose([
    transforms.RandomApply(torch.nn.ModuleList(
        [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05),
                                 interpolation=InterpolationMode.BILINEAR)]),
        p=0.5),
]),
    'woundhealing': transforms.Compose([transforms.RandomApply(
        torch.nn.ModuleList([transforms.RandomAffine(degrees=(-5, 5), translate=(0.01, 0.01),
                                                     interpolation=InterpolationMode.BILINEAR),
                             transforms.RandomHorizontalFlip(p=1),
                             transforms.RandomVerticalFlip(p=1)]), p=0.5)


    ])}


class loader2D(Dataset):
    def __init__(self, args):

        if args.run_mode == 'train':
            self.demo = pd.read_csv(args.csv_file_train, index_col=0)
            self.augmentation = True
            assert set(['fname', 'subject', args.targetname]).issubset(
                set(list(self.demo.columns))), f"Check input csv file column names"
        elif args.run_mode == 'val':
            self.demo = pd.read_csv(args.csv_file_val, index_col=0)
            self.augmentation = False
            assert set(['fname', 'subject', args.targetname]).issubset(
                set(list(self.demo.columns))), f"Check input csv file column names"
        else:
            self.demo = pd.read_csv(args.csv_file_test, index_col=0)
            self.augmentation = False
            assert set(['fname', 'subject', args.targetname]).issubset(
                set(list(self.demo.columns))), f"Check input csv file column names"

        # image directory
        IDunq = np.unique(self.demo['subject'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(self.demo['subject'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        img_height, img_width = args.image_size
        self.targetname = args.targetname
        self.imgdir = args.image_directory

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.index_combination = index_combination

        if self.augmentation:
            if args.jobname in dict_transform.keys():
                self.transform = dict_transform[args.jobname]
            else:
                self.transform = transforms.Compose([
                    transforms.RandomApply(torch.nn.ModuleList(
                        [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05, 0.05),
                                                 interpolation=InterpolationMode.BILINEAR)]),
                        p=0.5),
                ])

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]

        img1 = Image.open(os.path.join(self.imgdir, self.demo.fname[index1]))
        img1 = self.resize(img1)  # to tensor
        img2 = Image.open(os.path.join(self.imgdir, self.demo.fname[index2]))
        img2 = self.resize(img2)  # to tensor

        if self.augmentation:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return [np.array(img1), target1], [np.array(img2), target2]

    def __len__(self):
        return len(self.index_combination)


class loader3D(Dataset):
    def __init__(self, args):

        if args.run_mode == 'train':
            self.demo = pd.read_csv(args.csv_file_train, index_col=0)
            self.augmentation = True
            assert set(['fname', 'subject', args.targetname]).issubset(
                set(list(self.demo.columns))), f"Check input csv file column names"
        elif args.run_mode == 'val':
            self.demo = pd.read_csv(args.csv_file_val, index_col=0)
            self.augmentation = False

            assert set(['fname', 'subject', args.targetname]).issubset(
                set(list(self.demo.columns))), f"Check input csv file column names"
        else:
            self.demo = pd.read_csv(args.csv_file_test, index_col=0)
            self.augmentation = False
            assert set(['fname', 'subject', args.targetname]).issubset(
                set(list(self.demo.columns))), f"Check input csv file column names"

        self.jobname = args.jobname
        # Filter NaN values
        nan_indices = np.concatenate([np.where(np.isnan(self.demo[k]))[0] for k in [args.targetname] + args.optional_meta])
        if nan_indices.size>0:
            self.demo = self.demo.drop(index = nan_indices).reset_index(drop=True)

        # image directory
        IDunq = np.unique(self.demo['subject'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(self.demo['subject'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        self.image_size = args.image_size
        # TODO self.fnames = np.array('I' + meta.IMAGEUID.astype('int').astype('str') + '_mni_norm.nii.gz')

        self.targetname = args.targetname
        self.imgdir = args.image_directory
        self.index_combination = index_combination

        if len(args.optional_meta)>0:
            self.optional_meta = True
            self.optional_meta = np.array(self.demo[args.optional_meta]) # test  + args.optional_meta

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index].astype('int')
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]

        meta1, meta2 = self.optional_meta[index1, :], self.optional_meta[index2, :]

        fname1 = os.path.join(self.imgdir, self.fnames[int(index1)])
        fname2 = os.path.join(self.imgdir, self.fnames[int(index2)])

        image1 = tio.ScalarImage(fname1)
        image2 = tio.ScalarImage(fname2)

        resize = tio.transforms.Resize(tuple(self.image_size))
        image1 = resize(image1)
        image2 = resize(image2)

        if self.augmentation:
            pairwise_transform_list = []
            imagewise_transform_list = []

            if np.random.randint(0, 2):

                if not self.jobname == 'oasis-aging':# oasis-aging dataset w/o affine transform
                    if np.random.randint(0, 2):
                        affine_degree = tuple(np.random.uniform(low=-40, high=40, size=3))
                        affine_translate = tuple(np.random.uniform(low=-10, high=10, size=3))
                        pairwise_transform_list.append(tio.Affine(scales=(1, 1, 1),
                                                                  degrees=affine_degree,
                                                                  translation=affine_translate,
                                                                  image_interpolation='linear',
                                                                  default_pad_value='minimum'))

                if np.random.randint(0, 2):
                    pairwise_transform_list.append(tio.Flip(axes=('LR',)))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomNoise(mean=0, std=2))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomGamma(0.3))

            if np.random.randint(0, 2):
                imagewise_transform_list.append(tio.RandomBlur(2))

            if len(pairwise_transform_list) > 0:
                pairwise_augmentation = tio.Compose(pairwise_transform_list)
                image1 = pairwise_augmentation(image1)
                image2 = pairwise_augmentation(image2)

            if len(imagewise_transform_list) > 0:
                imagewise_augmentation = tio.Compose(imagewise_transform_list)
                image1 = imagewise_augmentation(image1)
                image2 = imagewise_augmentation(image2)

        image1 = image1.numpy().astype('float')
        image2 = image2.numpy().astype('float')

        if self.optional_meta:
            return [image1, target1, meta1], \
                   [image2, target2, meta2]

        else:
            return [image1, target1], \
                   [image2, target2]

    def __len__(self):
        return len(self.index_combination)


