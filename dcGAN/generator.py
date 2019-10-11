#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-11 10:06:44
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch import nn

class Generator(nn.Module):
	def __init__(self,h,w,c,latent_dim):
		super(Generator,self).__init__()
		self.h=h
		self.c=c
		self.w=w

		self.init_size=h//2**4
		self.pre_trans=nn.Linear(latent_dim,512*self.init_size**2)
		self.model=self.generator()
		
	def block(self,in_dim,out_dim):
		layers=nn.Sequential(
			nn.Conv2d(in_dim,out_dim,3,1,1),
			nn.BatchNorm2d(out_dim,0.8),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Upsample(scale_factor=2)
			)
		return layers
	def generator(self):
		model=nn.Sequential(
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Upsample(scale_factor=2),
			self.block(512,256),
			self.block(256,128),
			self.block(128,64),
			nn.Conv2d(64,self.c,3,1,1),
			nn.Tanh())
		return model
	def forward(self,z):
		latent_img=self.pre_trans(z)
		latent_img=latent_img.view(latent_img.size(0),512,self.init_size,self.init_size)
		img=self.model(latent_img)
		return img

if __name__=="__main__":
	g=Generator(28,28,1,100)