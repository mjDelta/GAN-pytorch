#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-23 16:31:53
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
from torch import autograd
import itertools
from torch.autograd import Variable

cuda=True if torch.cuda.is_available() else False
device=torch.device("cuda" if cuda else "cpu")
Tensor=torch.cuda.FloatTensor if cuda else torch.FloatTensor

class ReplayBuffer:
	def __init__(self,max_size=10):
		self.max_size=max_size
		self.data=[]
	def push_and_pop(self,data):
		returns=[]
		for e in data.data:
			e=torch.unsqueeze(e,0)
			if len(self.data)<self.max_size:
				self.data.append(e)
				returns.append(e)
			else:
				if np.random.uniform(0,1)>0.5:
					idx=np.random.randint(0,self.max_size-1)
					returns.append(self.data[idx].clone())
					self.data[idx]=e
				else:
					returns.append(e)
		return Variable(torch.cat(returns,dim=0))

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

h=256
w=256
c=3
batch_size=16
lr=0.0001
beta1=0.5
beta2=0.999
epochs=200
save_imgs=100
lambda_cyc=10.
lambda_id=5.
dataset="monet2photo"
out_dir="results"
generator_AB=Generator(l=2,n_filters=8)
generator_BA=Generator(l=2,n_filters=8)
discriminator_A=Discriminator(h,w,c)
discriminator_B=Discriminator(h,w,c)
gan_loss=nn.MSELoss()
cycle_loss=nn.L1Loss()
ident_loss=nn.L1Loss()
generator_AB.apply(weight_init)
generator_BA.apply(weight_init)
discriminator_A.apply(weight_init)
discriminator_B.apply(weight_init)
if cuda:
	generator_AB=generator_AB.cuda()
	generator_BA=generator_BA.cuda()
	discriminator_A=discriminator_A.cuda()
	discriminator_B=discriminator_B.cuda()
	gan_loss.cuda()
	cycle_loss.cuda()
	ident_loss.cuda()

os.makedirs(out_dir,exist_ok=True)
patch=(1,h//2**4,w//2**4)
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
	ImageDataset("../data/{}".format(dataset),transforms_=transforms_,mode="test"),
	batch_size=10,
	shuffle=True,
	)

optimizer_g=torch.optim.Adam(itertools.chain(generator_AB.parameters(),generator_BA.parameters()),lr=lr,betas=(beta1,beta2))
optimizer_d_A=torch.optim.Adam(discriminator_A.parameters(),lr=lr,betas=(beta1,beta2))
optimizer_d_B=torch.optim.Adam(discriminator_B.parameters(),lr=lr,betas=(beta1,beta2))

buffer_fake_A=ReplayBuffer(max_size=batch_size)
buffer_fake_B=ReplayBuffer(max_size=batch_size)

for e in range(epochs):
	identity_epoch_loss=0.
	gan_epoch_loss=0.
	cycle_epoch_loss=0.
	d_epoch_loss=0.
	cnter=0
	for b,(img_A,img_B) in enumerate(train_dataloader):

		real_A=torch.FloatTensor(img_A).to(device)
		real_B=torch.FloatTensor(img_B).to(device)

		valid=Tensor(np.ones((real_A.size(0),*patch)))
		fake=Tensor(np.zeros((real_A.size(0),*patch)))

		##train generator
		optimizer_g.zero_grad()

		fake_B=generator_AB(real_A)
		fake_A=generator_BA(real_B)

		gan_loss_B=gan_loss(discriminator_B(fake_B),valid)
		gan_loss_A=gan_loss(discriminator_A(fake_A),valid)
		gan_loss_=gan_loss_A+gan_loss_B

		ident_loss_A=ident_loss(generator_BA(real_A),real_A)
		ident_loss_B=ident_loss(generator_AB(real_B),real_B)
		ident_loss_=(ident_loss_A+ident_loss_B)*lambda_id

		recon_A=generator_BA(fake_B)
		recon_B=generator_AB(fake_A)
		cycle_loss_A=cycle_loss(real_A,recon_A)
		cycle_loss_B=cycle_loss(real_B,recon_B)
		cycle_loss_=(cycle_loss_A+cycle_loss_B)*lambda_cyc

		loss_G=gan_loss_+ident_loss_+cycle_loss_
		loss_G/=2
		loss_G.backward()
		optimizer_g.step()

		##train discriminator
		discriminator_A.zero_grad()
		loss_real_A=gan_loss(discriminator_A(real_A),valid)
		buffer_fake_As=buffer_fake_A.push_and_pop(fake_A)
		loss_fake_A=gan_loss(discriminator_A(buffer_fake_As),fake)
		loss_D_A=loss_real_A+loss_fake_A
		loss_D_A.backward()
		optimizer_d_A.step()

		discriminator_B.zero_grad()
		loss_real_B=gan_loss(discriminator_B(real_B),valid)
		buffer_fake_Bs=buffer_fake_B.push_and_pop(fake_B)
		loss_fake_B=gan_loss(discriminator_B(buffer_fake_Bs),fake)
		loss_D_B=loss_real_B+loss_fake_B
		loss_D_B.backward()
		optimizer_d_B.step()

		identity_epoch_loss+=ident_loss_.item()
		gan_epoch_loss+=gan_loss_.item()
		cycle_epoch_loss+=cycle_loss_.item()
		d_epoch_loss+=loss_D_A.item()
		d_epoch_loss+=loss_D_B.item()
		cnter+=1

		if (e*len(train_dataloader)//batch_size+b)%save_imgs==0:
			sample_images(e*len(train_dataloader)//batch_size+b,val_dataloader,generator_AB)
	print("epoch {}: d_loss {}\tcycle_loss {}\tidentity_loss {}\tgan_loss {}".format(e,round(d_epoch_loss/cnter,2),round(cycle_epoch_loss/cnter,2),round(identity_epoch_loss/cnter,2),round(gan_epoch_loss/cnter,2)))
			


