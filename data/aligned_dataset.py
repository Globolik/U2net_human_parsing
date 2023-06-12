from data.base_dataset import BaseDataset, Rescale_fixed, Normalize_image
from data.image_folder import make_dataset, make_dataset_test

import os
# import cv2
# import json
# import itertools
# import collections
# from tqdm import tqdm

import pandas as pd
import numpy as np
from PIL import Image
import cv2
import torch
import torchvision.transforms as transforms
import albumentations as A
from U2net_for_deep_fashion import *
class Deep_fashion_original_dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.image_folders_root = opt.image_folders_root
        else:
            self.image_folders_root = opt.image_folders_root_validation
        self.person_im_fodler = os.path.join(self.image_folders_root, 'image')
        self.mask_im_fodler = os.path.join(self.image_folders_root, 'image-parse-v3')
        self.names = pd.Series([_.split('.')[0] for _ in os.listdir(self.person_im_fodler)]).sample(frac=1, random_state=np.random.randint(0, 1000)).tolist()

        self.transform_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std)
            ])

    def __getitem__(self, index):
        # load images ad masks
        idx = index
        name = self.names[idx]
        img_path = os.path.join(self.person_im_fodler, name + '.jpg')
        img = Image.open(img_path)
        # TODO remove if wan to keep resolution
        # img = img.resize((self.width, self.height))
        img = np.array(img)


        mask_path = os.path.join(self.mask_im_fodler, name + '.png')
        mask = Image.open(mask_path)
        # TODO remove if wan to keep resolution
        # mask = mask.resize((self.width, self.height))
        mask = np.array(mask)

        mask_tensor = torch.as_tensor(mask, dtype=torch.int64)
        image_tensor = self.transform_rgb(img)





        return image_tensor, mask_tensor
    def reassign_labels_of_deepsashion(self, mask_arr):
        # inint with 0 and skip this label later
        new_mask = np.zeros(mask_arr.shape)

        for label_dep_f,labl_lip_new in enumerate(self.reassign_labels_info_new_labels):
            # starting from label 1 bc 'reassign_labels_info_new_labels' has no 0
            label_dep_f +=1
            if label_dep_f==self.skip_label_deep:
                continue
            new_mask[mask_arr==label_dep_f] = labl_lip_new
        return new_mask.astype('uint8')

    def __len__(self):
        return len(self.names)

    def name(self):
        return "Deep_fashion"
class Deep_fashion(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.image_folders_root = opt.image_folders_root
        else:
            self.image_folders_root = opt.image_folders_root_validation
        self.person_im_fodler = os.path.join(self.image_folders_root, 'image')
        self.mask_im_fodler = os.path.join(self.image_folders_root, 'image-parse-v3')
        self.names = pd.Series([_.split('.')[0] for _ in os.listdir(self.person_im_fodler)]).sample(frac=1, random_state=np.random.randint(0, 1000)).tolist()

        # for i_deepfasion - index in list
        # 0 label is the same
        self.reassign_labels_info = [5, 7, 12, 6, 9, 9, 1, 4, 11, 24, 18, 22, 2, 13, 10, 10, 10, 8, 10, 10, 23, 13, 5]
        self.skip_label_lip = 11
        self.skip_label_deep = 9

        self.sorted_lip_labels = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 18, 22, 23, 24, self.skip_label_lip]
        # convert to new labels that are continuous 0-15
        self.reassign_labels_info_new_labels = [self.sorted_lip_labels.index(_) for _ in self.reassign_labels_info]

        self.width = int(ORIG_WIDTH * SCALE_IM_SIZE)
        self.height = int(ORIG_HEIGHT * SCALE_IM_SIZE)


        self.transform_rgb = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=(0.9, 1.2), contrast=0.1, saturation=0.1, hue=0.1),
                    ],
                    p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std)
            ])
        self.transform_rgb_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std),
            ])

    def __getitem__(self, index):
        # load images ad masks
        idx = index
        name = self.names[idx]
        img_path = os.path.join(self.person_im_fodler, name + '.jpg')
        img = Image.open(img_path)
        # TODO remove if wan to keep resolution
        # img = img.resize((self.width, self.height))
        img = np.array(img)


        mask_path = os.path.join(self.mask_im_fodler, name + '.png')
        mask = Image.open(mask_path)
        # TODO remove if wan to keep resolution
        # mask = mask.resize((self.width, self.height))
        mask = np.array(mask)

        if self.isTrain:
            # outs = self.transform_augment(image=img, mask=mask)
            # img = outs['image']
            # mask = outs['mask']
            image_tensor = self.transform_rgb(img)

        else:
            image_tensor = self.transform_rgb_val(img)

        mask_tensor = torch.as_tensor(mask, dtype=torch.int64)




        return image_tensor, mask_tensor
    def reassign_labels_of_deepsashion(self, mask_arr):
        # inint with 0 and skip this label later
        new_mask = np.zeros(mask_arr.shape)

        for label_dep_f,labl_lip_new in enumerate(self.reassign_labels_info_new_labels):
            # starting from label 1 bc 'reassign_labels_info_new_labels' has no 0
            label_dep_f +=1
            if label_dep_f==self.skip_label_deep:
                continue
            new_mask[mask_arr==label_dep_f] = labl_lip_new
        return new_mask.astype('uint8')

    def __len__(self):
        return len(self.names)

    def name(self):
        return "Deep_fashion"
class Deep_fashion_11_labels(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.image_folders_root = opt.image_folders_root
        else:
            self.image_folders_root = opt.image_folders_root_validation
        self.person_im_fodler = os.path.join(self.image_folders_root, 'image')
        self.mask_im_fodler = os.path.join(self.image_folders_root, 'image-parse-v3')
        self.names = pd.Series([_.split('.')[0] for _ in os.listdir(self.person_im_fodler)]).sample(frac=1, random_state=np.random.randint(0, 1000)).tolist()


        self.width = int(ORIG_WIDTH * SCALE_IM_SIZE)
        self.height = int(ORIG_HEIGHT * SCALE_IM_SIZE)


        self.transform_rgb = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=(0.9, 1.2), contrast=0.1, saturation=0.1, hue=0.1),
                    ],
                    p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([opt.mean,opt.mean,opt.mean], [opt.std,opt.std,opt.std])
            ])
        self.transform_rgb_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([opt.mean,opt.mean,opt.mean], [opt.std,opt.std,opt.std]),
            ])

        self.transform_augment = A.Compose([
            A.HorizontalFlip(p=0.5)
        ])
    def __getitem__(self, index):
        # load images ad masks
        idx = index
        name = self.names[idx]
        img_path = os.path.join(self.person_im_fodler, name + '.jpg')
        img = Image.open(img_path)
        # TODO remove if wan to keep resolution
        # img = img.resize((self.width, self.height))
        img = np.array(img)


        mask_path = os.path.join(self.mask_im_fodler, name + '.png')
        mask = Image.open(mask_path)
        # TODO remove if wan to keep resolution
        # mask = mask.resize((self.width, self.height))
        mask = np.array(mask)

        if self.isTrain:
            outs = self.transform_augment(image=img, mask=mask)
            img = outs['image']
            mask = outs['mask']
            image_tensor = self.transform_rgb(img)

        else:
            image_tensor = self.transform_rgb_val(img)

        mask_tensor = torch.as_tensor(mask, dtype=torch.int64)




        return image_tensor, mask_tensor
    def __len__(self):
        return len(self.names)

    def name(self):
        return "Deep_fashion"
class Deep_fashion_9_labels(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.image_folders_root = opt.image_folders_root
        else:
            self.image_folders_root = opt.image_folders_root_validation
        self.person_im_fodler = os.path.join(self.image_folders_root, 'image')
        self.mask_im_fodler = os.path.join(self.image_folders_root, 'image-parse-v3')
        self.names = pd.Series([_.split('.')[0] for _ in os.listdir(self.person_im_fodler)]).sample(frac=1, random_state=np.random.randint(0, 1000)).tolist()


        self.width = int(ORIG_WIDTH * SCALE_IM_SIZE)
        self.height = int(ORIG_HEIGHT * SCALE_IM_SIZE)


        self.transform_rgb = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=(0.9, 1.2), contrast=0.1, saturation=0.1, hue=0.1),
                    ],
                    p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std)
            ])
        self.transform_rgb_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std),
            ])
    def __getitem__(self, index):
        # load images ad masks
        idx = index
        name = self.names[idx]
        img_path = os.path.join(self.person_im_fodler, name + '.jpg')
        img = Image.open(img_path)
        # TODO remove if wan to keep resolution
        # img = img.resize((self.width, self.height))
        img = np.array(img)


        mask_path = os.path.join(self.mask_im_fodler, name + '.png')
        mask = Image.open(mask_path)
        # TODO remove if wan to keep resolution
        # mask = mask.resize((self.width, self.height))
        mask = np.array(mask)

        if self.isTrain:
            image_tensor = self.transform_rgb(img)

        else:
            image_tensor = self.transform_rgb_val(img)

        mask_tensor = torch.as_tensor(mask, dtype=torch.int64)




        return image_tensor, mask_tensor
    def __len__(self):
        return len(self.names)

    def name(self):
        return "Deep_fashion"

class Imat_19(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.image_folders_root = opt.image_folders_root
        else:
            self.image_folders_root = opt.image_folders_root_validation
        self.person_im_fodler = os.path.join(self.image_folders_root, 'image')
        self.mask_im_fodler = os.path.join(self.image_folders_root, 'image-parse-v3')
        self.names = pd.Series([_.split('.')[0] for _ in os.listdir(self.person_im_fodler)]).sample(frac=1, random_state=np.random.randint(0, 1000)).tolist()

        self.label_pairs = [(11, 12), (15, 16), (17, 18)]
        self.width = int(ORIG_WIDTH * SCALE_IM_SIZE)
        self.height = int(ORIG_HEIGHT * SCALE_IM_SIZE)


        self.transform_rgb = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=(0.9, 1.2), contrast=0.1, saturation=0.1, hue=0.1),],
                    p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std)
            ])
        self.transform_rgb_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std),
            ])

        self.transform_augment_hf = A.Compose([
            A.HorizontalFlip(always_apply=True),
        ])
        self.transform_augment_jpeg = A.Compose([
            A.OneOf([
                A.JpegCompression(quality_lower=50, quality_upper=70, p=0.6),
                A.Defocus(radius=(2, 3), p=0.4),
            ], p=0.3),
        ])
        self.transform_augment_rotate = A.Compose([
            A.Rotate(limit=30, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
                     crop_border=False, p=0.2),
        ])


    def __getitem__(self, index):
        # load images ad masks
        idx = index
        name = self.names[idx]
        img_path = os.path.join(self.person_im_fodler, name + '.jpg')
        img = Image.open(img_path)
        # TODO remove if wan to keep resolution
        # img = img.resize((self.width, self.height))
        img = np.array(img)


        mask_path = os.path.join(self.mask_im_fodler, name + '.png')
        mask = Image.open(mask_path)
        # TODO remove if wan to keep resolution
        # mask = mask.resize((self.width, self.height))
        mask = np.array(mask)

        if self.isTrain:
            img = self.transform_augment_jpeg(image=img)['image']
            aug = self.transform_augment_rotate(image=img, mask=mask)
            img, mask = aug['image'], aug['mask']


            if np.random.rand()>0.5:
                aug = self.transform_augment_hf(image=img, mask=mask)
                img = aug['image']
                mask = aug['mask']
                for l, r in self.label_pairs:
                    l_mask = mask == l
                    r_mask = mask == r
                    mask[l_mask] = r
                    mask[r_mask] = l

            image_tensor = self.transform_rgb(img)

        else:
            image_tensor = self.transform_rgb_val(img)
        mask_tensor = torch.as_tensor(mask, dtype=torch.int64)
        return image_tensor, mask_tensor


    def __len__(self):
        return len(self.names)

    def name(self):
        return "Deep_fashion"
class Imat_19_QAT(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if self.isTrain:
            self.image_folders_root = opt.image_folders_root
        else:
            self.image_folders_root = opt.image_folders_root_validation
        self.person_im_fodler = os.path.join(self.image_folders_root, 'image')
        self.mask_im_fodler = os.path.join(self.image_folders_root, 'image-parse-v3')
        self.names = pd.Series([_.split('.')[0] for _ in os.listdir(self.person_im_fodler)]).sample(frac=1, random_state=np.random.randint(0, 1000))

        self.names = self.names.sample(n=opt.NUMBER_OF_ITEMS_TO_USE).tolist()

        self.label_pairs = [(11, 12), (15, 16), (17, 18)]
        self.width = int(ORIG_WIDTH * SCALE_IM_SIZE)
        self.height = int(ORIG_HEIGHT * SCALE_IM_SIZE)


        self.transform_rgb = transforms.Compose(
            [
                transforms.ToPILImage(),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(brightness=(0.9, 1.2), contrast=0.1, saturation=0.1, hue=0.1),],
                #     p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std)
            ])
        self.transform_rgb_val = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(opt.mean, opt.std),
            ])

        self.transform_augment_hf = A.Compose([
            A.HorizontalFlip(always_apply=True),
        ])
        # self.transform_augment_jpeg = A.Compose([
        #     A.OneOf([
        #         A.JpegCompression(quality_lower=50, quality_upper=70, p=0.6),
        #         A.Defocus(radius=(2, 3), p=0.4),
        #     ], p=0.3),
        # ])
        # self.transform_augment_rotate = A.Compose([
        #     A.Rotate(limit=30, interpolation=cv2.INTER_NEAREST, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0,
        #              crop_border=False, p=0.2),
        # ])


    def __getitem__(self, index):
        # load images ad masks
        idx = index
        name = self.names[idx]
        img_path = os.path.join(self.person_im_fodler, name + '.jpg')
        img = Image.open(img_path)
        # TODO remove if wan to keep resolution
        # img = img.resize((self.width, self.height))
        img = np.array(img)


        mask_path = os.path.join(self.mask_im_fodler, name + '.png')
        mask = Image.open(mask_path)
        # TODO remove if wan to keep resolution
        # mask = mask.resize((self.width, self.height))
        mask = np.array(mask)

        if self.isTrain:
            # img = self.transform_augment_jpeg(image=img)['image']
            # aug = self.transform_augment_rotate(image=img, mask=mask)
            # img, mask = aug['image'], aug['mask']


            if np.random.rand()>0.5:
                aug = self.transform_augment_hf(image=img, mask=mask)
                img = aug['image']
                mask = aug['mask']
                for l, r in self.label_pairs:
                    l_mask = mask == l
                    r_mask = mask == r
                    mask[l_mask] = r
                    mask[r_mask] = l

            image_tensor = self.transform_rgb(img)

        else:
            image_tensor = self.transform_rgb_val(img)
        mask_tensor = torch.as_tensor(mask, dtype=torch.int64)
        return image_tensor, mask_tensor


    def __len__(self):
        return len(self.names)

    def name(self):
        return "QAT_IMAT"

