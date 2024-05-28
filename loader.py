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

dict_transform = {}

class loader2D(Dataset):
    def __init__(self, root='/share/sablab/nfs04//data/embryo/', transform=None, trainvaltest='train', opt = None):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root)

        meta = pd.read_csv(os.path.join(root, 'demo.csv'), index_col=0)

        num_of_subjects = len(np.unique(meta['embryoname']))
        meta = meta[meta.trainvaltest == trainvaltest].reset_index()
        IDunq = np.unique(meta['embryoname'])
        index_combination = np.empty((0, 2))
        for sid in IDunq:
            indices = np.where(meta['embryoname'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)


        img_height, img_width = opt.image_size
        self.targetname = opt.targetname

        self.resize = transforms.Compose([
            transforms.Resize((img_height, img_width), InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.index_combination = index_combination
        self.transform = transform
        self.demo = meta

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]

        img1 = Image.open(os.path.join(self.imgdir, self.demo.fname[index1]))
        img1 = self.resize(img1)  # to tensor
        img2 = Image.open(os.path.join(self.imgdir, self.demo.fname[index2]))
        img2 = self.resize(img2)  # to tensor

        if self.transform:
            augmentation = transforms.Compose([
                transforms.RandomApply(torch.nn.ModuleList(
                    [transforms.RandomAffine(degrees=(-10, 10), translate=(0.05,0.05),
                                             interpolation=InterpolationMode.BILINEAR)]),
                    p=0.5),
            ])

            img1 = augmentation(img1)
            img2 = augmentation(img2)

        return [np.array(img1), target1], [np.array(img2), target2]

    def __len__(self):
        return len(self.index_combination)


class loader3D(Dataset):
    def __init__(self, args):

         # root='/share/sablab/nfs04/data/ADNI_mci/', transform=None,
         # trainvaltest='train',opt=None, easy2hard_thr=0, positive_pairs_only=False):

        self.trainvaltest = trainvaltest
        self.imgdir = os.path.join(root, 'image/')
        self.targetname = opt.targetname
        self.transform = transform
        self.deltaT = opt.deltaT
        self.diffT = opt.diffT
        self.diffMMSE = opt.diffMMSE
        self.diffCDRSB = opt.diffCDRSB
        self.diffTSEX = opt.diffTSEX

        self.sex = opt.SEX
        self.easy2hard_thr = easy2hard_thr
        self.positive_pairs_only = positive_pairs_only
        self.baseage = opt.baseAGE

        # Preprocessed (using NPP)
        meta = pd.read_csv(os.path.join(root, 'demo', 'ADNIMERGE_MCI_long_dx_conversion.csv'), index_col=0,
                           low_memory=False)
        meta = meta[meta.trainvaltest == trainvaltest].reset_index(drop=True)
        # Sort demo
        meta = meta.sort_values(by=['PTID', 'Month']).reset_index(drop=True)
        IDunq = np.unique(meta['PTID'])
        index_combination = np.empty((0, 2))

        # target 'nan' filter
        meta = meta[~np.isnan(meta[opt.targetname])].reset_index(drop=True)
        if self.diffMMSE:
            meta = meta[~np.isnan(meta['MMSE'])].reset_index(drop=True)
        if self.diffCDRSB:
            meta = meta[~np.isnan(meta['CDRSB'])].reset_index(drop=True)

        for sid in IDunq:
            indices = np.where(meta['PTID'] == sid)[0]
            ### all possible pairs
            tmp_combination = np.array(
                np.meshgrid(np.array(range(len(indices))), np.array(range(len(indices))))).T.reshape(-1, 2)
            index_combination = np.append(index_combination, indices[tmp_combination], 0)

        # only one direction.
        if self.positive_pairs_only:
            index_combination = (index_combination[(index_combination[:, 1] - index_combination[:, 0]) > 0]).astype('int')

        if trainvaltest == 'train':
            target1 = meta.DX.iloc[index_combination[:, 0]]
            target2 = meta.DX.iloc[index_combination[:, 1]]
            targetdiff = np.array(target1) != np.array(target2) # NOTE: without these pairs its still odd and will output .6
            index_append = index_combination[np.where(targetdiff)[0], :]
            index_combination = np.append(index_combination, index_append, 0)
            index_combination = np.append(index_combination, index_append, 0)
            index_combination = np.append(index_combination, index_append, 0)

            # -- confirm that the pairs are temporally ordered
            # month = np.array(meta.Month)
            # assert np.sum((month[index_combination[:, 1]] - month[index_combination[:, 0]]) >= 0) == len(index_combination), "Subject ID and Month order invalid. Check Demo"

            month_diff = np.array(meta.M[index_combination[:, 1]]) - np.array(meta.M[index_combination[:, 0]])

            if self.easy2hard_thr>0:
                month_bool = np.abs(month_diff) > self.easy2hard_thr
                index_combination = index_combination[month_bool, :]
                print(f'Easy2hard: time difference threshold {self.easy2hard_thr}')

        self.index_combination = index_combination
        self.demo = meta
        self.image_size = opt.image_size
        self.fnames = np.array('I'+meta.IMAGEUID.astype('int').astype('str') + '_mni_norm.nii.gz')

    def __getitem__(self, index):
        index1, index2 = self.index_combination[index]
        target1, target2 = self.demo[self.targetname][index1], self.demo[self.targetname][index2]
        age1, age2 = self.demo['age'][index1], self.demo['age'][index2]

        if self.deltaT or self.diffT:
            score1, score2 = self.demo.Month[index1], self.demo.Month[index2] # todo: note: anyhow the delta will be used ("Age" in this demo is unreliable) : update: "age" is fine.
        if self.diffMMSE:
            score1, score2 = self.demo.MMSE[index1], self.demo.MMSE[index2]
        if self.diffCDRSB:
            score1, score2 = self.demo.CDRSB[index1], self.demo.CDRSB[index2]
        if self.sex:
            sex = np.array(self.demo.PTGENDER[index1] == 'Male').astype(int)
        if self.diffTSEX:
            sex_month = (np.array(self.demo.PTGENDER == 'Male').astype("int") * self.demo.Month)
            score1, score2 = sex_month[index1], sex_month[index2]

        fname1 = os.path.join(self.imgdir, self.fnames[int(index1)])
        fname2 = os.path.join(self.imgdir, self.fnames[int(index2)])

        image1 = tio.ScalarImage(fname1)
        image2 = tio.ScalarImage(fname2)

        resize = tio.transforms.Resize(tuple(self.image_size))
        image1 = resize(image1)
        image2 = resize(image2)

        if self.transform:
            # pairwise transform
            pairwise_transform_list = []
            imagewise_transform_list = []

            if np.random.randint(0, 2):
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

            # if np.random.randint(0, 2):
            #     imagewise_transform_list.append(tio.RandomSwap())

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

        if self.deltaT or self.diffT or self.diffMMSE or self.diffCDRSB or self.diffTSEX:

            if self.baseage:
                return [image1, target1, score1, age1], \
                       [image2, target2, score2, age2]

            elif self.sex:
                return [image1, target1, score1, sex], \
                       [image2, target2, score2, sex]

            else:
                return [image1, target1, score1], \
                       [image2, target2, score2]
        else:

            if self.baseage:
                return [image1, target1, age1], \
                       [image2, target2, age2]

            elif self.sex:
                return [image1, target1, sex], \
                    [image2, target2, sex]

            else:
                return [image1, target1], \
                       [image2, target2]

    def __len__(self):
        return len(self.index_combination)
