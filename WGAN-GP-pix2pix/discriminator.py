#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-21 19:35:26
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch import nn
import torch


class Discriminator(nn.Module):
	def __init__(self,h,w,c):
		super(Discriminator,self).__init__()
		self.h=h
		self.w=w
		self.conv=self.conv_block(c)
		self.linear=self.linear_block()

	def block(self,in_dim,out_dim):
		layer=nn.Sequential(
			nn.Conv2d(in_dim,out_dim,3,2,1),
			nn.LeakyReLU(0.2,inplace=True),
			)
		return layer
	def conv_block(self,c):
		model=nn.Sequential(
			self.block(c,64),#128
			self.block(64,128),#64
			self.block(128,256),#32
			self.block(256,256),#16
			self.block(256,256),#8
			self.block(256,256),#4
			self.block(256,256),#2
			)
		return model
	def linear_block(self):
		model=nn.Sequential(
			nn.Linear(1024,256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Linear(256,1)
			)
		return model

	def forward(self,img):
		validity=self.conv(img)
		validity_flat=validity.view(validity.size(0),-1)
		score=self.linear(validity_flat)
		return score
if __name__=="__main__":
	model=Discriminator(28,28,1)
