#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-11 20:20:47
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch.nn as nn
import torch
from generator import Generator
from discriminator import Discriminator
import os
from torchvision import datasets
from torchvision import transforms 
from torchvision.utils import save_image
import numpy as np
from datasets import *
cuda=True if torch.cuda.is_available() else False
device=torch.device("cuda" if cuda else "cpu")

def weight_init(m):
	if isinstance(m,nn.Conv2d):
		nn.init.xavier_normal_(m.weight.data)
		if m.bias is not None:
			nn.init.normal_(m.bias.data)
	elif isinstance(m,nn.BatchNorm2d):
		nn.init.normal_(m.weight.data,mean=1,std=0.02)
		nn.init.constant_(m.bias.data,0)
	elif isinstance(m,nn.Linear):
		nn.init.xavier_normal_(m.weight.data)
		nn.init.normal_(m.bias.data)
def sample_images(trained_batches,dataloader,generator):
	imgs_A,imgs_B=next(iter(dataloader))
	real_As=torch.FloatTensor(imgs_A).to(device)
	real_Bs=torch.FloatTensor(imgs_B).to(device)	
	fake_Bs=generator(real_As)
	imgs=torch.cat((real_As.data,real_Bs.data,fake_Bs.data),-2)
	save_image(imgs,"results/images{}.png".format(trained_batches),nrow=5,normalize=True)
h=256
w=256
c=3
batch_size=1
lr=0.0002
beta1=0.5
beta2=0.999
epochs=200
n_cpu=4
save_imgs=500
dataset="CMP_facade_DB_base"

generator=Generator(l=2)
discriminator=Discriminator(h,w,c)
adversarial_loss=nn.MSELoss()
pix_loss=nn.L1Loss()
if cuda:
	generator=generator.cuda()
	discriminator=discriminator.cuda()
	adversarial_loss=adversarial_loss.cuda()
	pix_loss=pix_loss.cuda()
generator.apply(weight_init)
discriminator.apply(weight_init)

os.makedirs("results",exist_ok=True)

transforms_=[
	transforms.Resize((h,w),Image.BICUBIC),
	transforms.ToTensor(),
	transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]

train_dataloader=torch.utils.data.DataLoader(
	ImageDataset("../data/{}".format(dataset),transforms_=transforms_),
	batch_size=batch_size,
	shuffle=True,
	)
val_dataloader=torch.utils.data.DataLoader(
	ImageDataset("../data/{}".format(dataset),transforms_=transforms_,mode="val"),
	batch_size=10,
	shuffle=True,
	)

optimizer_g=torch.optim.Adam(generator.parameters(),lr=lr,betas=(beta1,beta2))
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(beta1,beta2))

Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor

patch=(1,h//2**4,w//2**4)
lambda_p=100
for e in range(epochs):
	for b,(img_A,img_B) in enumerate(train_dataloader):

		valid=Tensor(img_A.size(0),*patch).fill_(1.0)
		fake=Tensor(img_A.size(0),*patch).fill_(0.0)

		real_A=torch.FloatTensor(img_A).to(device)
		real_B=torch.FloatTensor(img_B).to(device)

		##Train generator
		optimizer_g.zero_grad()
		fake_B=generator(real_A)
		pred_fake=discriminator(real_A,fake_B)
		g_loss=adversarial_loss(pred_fake,valid)
		p_loss=pix_loss(real_B,fake_B)
		g_total_loss=g_loss+lambda_p*p_loss
		g_total_loss.backward()
		optimizer_g.step()

		##Train discriminator
		optimizer_d.zero_grad()
		real_loss=adversarial_loss(discriminator(real_A,real_B),valid)
		fake_loss=adversarial_loss(discriminator(real_A,fake_B.detach()),fake)
		d_loss=(real_loss+fake_loss)/2
		d_loss.backward()
		optimizer_d.step()

		if (e*len(train_dataloader)+b)%save_imgs==0:
			sample_images(e*len(train_dataloader)+b,val_dataloader,generator)
			print("epoch {} batch {}: d_loss {} g_loss {} p_loss {}".format(e,b,round(d_loss.item(),2),round(g_loss.item(),2)),,round(p_loss.item(),2))
