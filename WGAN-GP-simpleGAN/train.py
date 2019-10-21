#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-21 10:39:27
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
from pytorch import autograd
cuda=True if torch.cuda.is_available() else False
device=torch.device("cuda" if cuda else "cpu")
Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor

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
def get_gradient_penalty(D,real_imgs,fake_imgs):
	alpha=Tensor(np.random.random((real_imgs.size(0),1,1,1)))
	interpolates=(alpha*real_imgs+(1-alpha)*fake_imgs).requires_grad_(True)
	d_interpolate=D(interpolates)
	fake=Tensor(real_imgs.size(0),1).fill_(1.0)
	gradients=autograd.grad(
		outputs=d_interpolate,
		inputs=interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True
		)[0]
	gradients=gradients.view(gradients.size(0),-1)
	gradients_penalty=(gradients.norm(2,dim=1)-1)**2
	gradients_penalty=gradients_penalty.mean()
	return gradients_penalty

h=28
w=28
batch_size=64
lr=0.0002
latent_dim=100
beta1=0.5
beta2=0.999
epochs=200
save_imgs=8
lambda_gp=10
n_critic=5
dataset="mnist"
out_dir="results"

generator=Generator(h,w,c,latent_dim)
discriminator=Discriminator(h,w,c)
if cuda:
	generator=generator.cuda()
	discriminator=discriminator.cuda()
generator.apply(weight_init)
discriminator.apply(weight_init)

os.makedirs(out_dir,exist_ok=True)
os.makedirs("../data/{}".format(dataset),exist_ok=True)
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
optimizer_d=torch.optim.RMSprop(discriminator.parameters(),lr=lr)

for e in range(epochs):
	for b,(imgs,_) in enumerate(dataloader):
		real_imgs=torch.FloatTensor(imgs).to(device)
		z=Tensor(np.random.normal(0,1,(real_imgs.shape[0],latent_dim)))
		fake_imgs=generator(z)

		##Train discriminator
		optimizer_d.zero_grad()
		real_score=discriminator(real_imgs)
		fake_score=discriminator(fake_imgs)
		gradient_penalty=get_gradient_penalty(discriminator,real_imgs,fake_imgs)
		d_loss=-torch.mean(real_score)+torch.mean(fake_score)+lambda_gp*gradient_penalty
		d_loss.backward()
		optimizer_d.step()

		if b%n_critic==0:

			##Train generator
			optimizer_g.zero_grad()
			fake_score_g=discriminator(fake_imgs)
			g_loss=-torch.mean(fake_score_g)
			g_loss.backward()
			optimizer_g.step()

		if (e*len(dataloader)+b)%save_imgs==0:
			save_image(fake_imgs.data[:25],"results/images{}.png".format(e*len(dataloader)+b),nrow=5,normalize=True)
	print("epoch {} batch {}: d_loss {} g_loss {}".format(e,b,round(d_loss,2),round(g_loss,2)))
			

