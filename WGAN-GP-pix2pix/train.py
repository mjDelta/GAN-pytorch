#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-21 19:36:23
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch.nn as nn
import torch
from generator import Generator2
from discriminator import Discriminator
import os
from torchvision import datasets
from torchvision import transforms 
from torchvision.utils import save_image
import numpy as np
from datasets import *
from torch import autograd

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
def sample_images(trained_batches,dataloader,generator):
	imgs_A,imgs_B=next(iter(dataloader))
	real_As=torch.FloatTensor(imgs_A).to(device)
	real_Bs=torch.FloatTensor(imgs_B).to(device)	
	fake_Bs=generator(real_As)
	imgs=torch.cat((real_As.data,real_Bs.data,fake_Bs.data),-2)
	save_image(imgs,"{}/images{}.png".format(out_dir,trained_batches),nrow=5,normalize=True)
def get_gradient_penalty(D,real,fake):
	alpha=Tensor(np.random.random((real.size(0),1,1,1)))
	interpolates=(alpha*real+(1-alpha)*fake).requires_grad_(True)
	d_interpolate=D(interpolates)
	fake=Tensor(real.size(0),1).fill_(1.0)
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
h=256
w=256
c=3
batch_size=4
lr=0.0001
beta1=0.9
beta2=0.999
epochs=200
save_imgs=30
lambda_p=50
lambda_gp=10
n_critic=5
dataset="CMP_facade_DB_base"
out_dir="results7-seg2img"
generator=Generator2(n_filters=32,kernel_size=3,l=4)
discriminator=Discriminator(h,w,c)
pix_loss=nn.MSELoss()
if cuda:
	generator=generator.cuda()
	discriminator=discriminator.cuda()
	pix_loss=pix_loss.cuda()
generator.apply(weight_init)
discriminator.apply(weight_init)

os.makedirs(out_dir,exist_ok=True)

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

for e in range(epochs):
	p_epoch_loss=0.
	g_epoch_loss=0.
	d_epoch_loss=0.
	for b,(img_A,img_B) in enumerate(train_dataloader):

		real_A=torch.FloatTensor(img_A).to(device)
		real_B=torch.FloatTensor(img_B).to(device)
		fake_B=generator(real_A)

		##Train discriminator
		optimizer_d.zero_grad()
		real_score=discriminator(real_B)
		fake_score=discriminator(fake_B.detach())
		gradient_penalty=get_gradient_penalty(discriminator,real_B.data,fake_B.data)
		d_loss=-torch.mean(real_score)+torch.mean(fake_score)+lambda_gp*gradient_penalty
		d_epoch_loss+=d_loss.item()
		d_loss.backward()
		optimizer_d.step()
		if b%n_critic==0:
			##Train generator
			optimizer_g.zero_grad()
			# fake_B=generator(real_A)
			fake_score_g=discriminator(fake_B)
			p_loss=lambda_p*pix_loss(real_B,fake_B)
			g_loss=-torch.mean(fake_score_g)
			g_total_loss=g_loss+p_loss
			p_epoch_loss+=p_loss.item()
			g_epoch_loss+=g_loss.item()
			g_total_loss.backward()
			optimizer_g.step()



		if (e*len(train_dataloader)+b)%save_imgs==0:
			sample_images(e*len(train_dataloader)+b,val_dataloader,generator)
	print("epoch {} batch {}: d_loss {} g_loss {} p_loss {}".format(e,b,round(d_epoch_loss/len(train_dataloader),2),round(g_epoch_loss/len(train_dataloader),2),round(p_epoch_loss/len(train_dataloader),2)))
			

