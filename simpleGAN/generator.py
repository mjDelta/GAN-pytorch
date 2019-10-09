#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-09 23:56:06
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch import nn

class Generator(nn.Module):
	def __init__(self,h,w,c,latent_dim):
		super(Generator,self).__init__()
		self.h=h
		self.c=c
		self.w=w

		self.model=self.generator(h,w,c,latent_dim)

	def block(self,in_dim,out_dim):
		layers=nn.Sequential(
			nn.Linear(in_dim,out_dim),
			nn.BatchNorm1d(out_dim,0.8),
			nn.LeakyReLU(0.2,inplace=True)
			)
		return layers
	def generator(self,h,w,c,latent_dim):
		model=nn.ModuleList(
			[self.block(latent_dim,128),
			self.block(128,256),
			self.block(256,512),
			self.block(512,1024),
			nn.Linear(1024,h*w*c),
			nn.Tanh()])
		return model
	def forward(self,z):
		img=self.model(z)
		img=img.view(img.size(0),(self.c,self.h,self.w))
		return img

if __name__=="__main__":
	g=Generator(28,28,1,100)
