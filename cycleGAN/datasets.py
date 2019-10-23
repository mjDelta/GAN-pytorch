#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-23 16:29:28
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

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

        self.A_files = sorted(glob.glob(os.path.join(root, mode+"A") + "/*.*"))
        self.B_files = sorted(glob.glob(os.path.join(root, mode+"B") + "/*.*"))


    def __getitem__(self, index):
        img_A = Image.open(self.A_files[index % len(self.A_files)])
        img_B = Image.open(self.B_files[index % len(self.B_files)])

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A,img_B
        # return img_B,img_A

    def __len__(self):
        return max(len(self.A_files),len(self.B_files))

if __name__=="__main__":
	data=ImageDataset("../data/monet2photo")
	print(len(data))