#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-11 20:29:47
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)
# modified from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/pix2pix/datasets.py
import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        self.transform = transforms.Compose(transforms_)

        self.A_files = sorted(glob.glob(os.path.join(root, mode) + "/*.jpg"))
        self.B_files = sorted(glob.glob(os.path.join(root, mode) + "/*.png"))


    def __getitem__(self, index):
        img_A = Image.open(self.A_files[index % len(self.A_files)])
        img_B = Image.open(self.B_files[index % len(self.B_files)]).convert("RGB")
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A,img_B

    def __len__(self):
        return len(self.B_files)