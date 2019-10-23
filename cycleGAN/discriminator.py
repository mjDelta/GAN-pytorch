#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-23 16:25:48
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
			self.block(c,32),
			self.block(32,64),
			self.block(64,128),
			self.block(128,256),
			nn.Conv2d(256,c,3,1,1)
			)
		return model

	def forward(self,img):
		validity=self.model(img)
		return validity
if __name__=="__main__":
	model=Discriminator(28,28,1)
