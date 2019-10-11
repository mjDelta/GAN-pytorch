#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-11 10:31:59
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
cuda=True if torch.cuda.is_available() else False
device=torch.device("cuda" if cuda else "cpu")

h=32
w=32
c=1
latent_dim=128
batch_size=64
lr=0.0002
beta1=0.5
beta2=0.999
epochs=200

save_imgs=500
generator=Generator(h,w,c,latent_dim)
discriminator=Discriminator(h,w,c)
adversarial_loss=nn.BCELoss()

if cuda:
	generator=generator.cuda()
	discriminator=discriminator.cuda()
	adversarial_loss=adversarial_loss.cuda()

os.makedirs("../data/mnist",exist_ok=True)
os.makedirs("results",exist_ok=True)
dataloader=torch.utils.data.DataLoader(
	datasets.MNIST(
		"../data/mnist",
		train=True,
		download=True,
		transform=transforms.Compose(
			[transforms.Resize(h),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])]##[mean,std]
			)
		),
	batch_size=batch_size,
	shuffle=True
	)

optimizer_g=torch.optim.Adam(generator.parameters(),lr=lr,betas=(beta1,beta2))
optimizer_d=torch.optim.Adam(discriminator.parameters(),lr=lr,betas=(beta1,beta2))

Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor

for e in range(epochs):
	for b,(imgs,_) in enumerate(dataloader):

		valid=Tensor(imgs.size(0),1).fill_(1.0)
		fake=Tensor(imgs.size(0),1).fill_(0.0)

		real_imgs=torch.FloatTensor(imgs).to(device)

		z=Tensor((np.random.normal(0,1,(imgs.shape[0],latent_dim))))
		gen_imgs=generator(z)

		##Train generator
		optimizer_g.zero_grad()
		g_loss=adversarial_loss(discriminator(gen_imgs),valid)
		g_loss.backward()
		optimizer_g.step()

		##Train discriminator
		optimizer_d.zero_grad()
		real_loss=adversarial_loss(discriminator(real_imgs),valid)
		fake_loss=adversarial_loss(discriminator(gen_imgs.detach()),fake)
		d_loss=(real_loss+fake_loss)/2
		d_loss.backward()
		optimizer_d.step()

		if (e*len(dataloader)+b)%save_imgs==0:
			save_image(gen_imgs.data[:25],"results/images{}.png".format(e*len(dataloader)+b),nrow=5,normalize=True)
			print("epoch {} batch {}: d_loss {} g_loss {}".format(e,b,round(d_loss.item(),2),round(g_loss.item(),2)))