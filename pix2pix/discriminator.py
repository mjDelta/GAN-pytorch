#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-11 19:22:11
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch import nn
import torch


class Discriminator(nn.Module):
	def __init__(self,h,w,c):
		super(Discriminator,self).__init__()

		self.h=h
		self.w=w
		self.model=self.discrminator(c)


	def block(self,in_dim,out_dim):
		layer=nn.Sequential(
			nn.Conv2d(in_dim,out_dim,3,2,1),
			nn.LeakyReLU(0.2,inplace=True),
			nn.BatchNorm2d(out_dim)
			)
		return layer
	def discrminator(self,c):
		model=nn.Sequential(
			self.block(c*2,64),
			self.block(64,128),
			self.block(128,256),
			self.block(256,512),
			nn.Conv2d(512,1,3,1,1)
			)
		return model

	def forward(self,img_A,img_B):
		img=torch.cat((img_A,img_B),1)
		validity=self.model(img)

		return validity
if __name__=="__main__":
	model=Discriminator(28,28,1)