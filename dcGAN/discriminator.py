#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-11 10:23:49
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch import nn

class Discriminator(nn.Module):
	def __init__(self,h,w,c):
		super(Discriminator,self).__init__()

		self.h=h
		self.w=w
		self.model=self.discrminator(c)
		self.clf_layer=nn.Sequential(
			nn.Linear(512*(self.h//2**4)**2,1),
			nn.Sigmoid()
			)

	def block(self,in_dim,out_dim):
		layer=nn.Sequential(
			nn.Conv2d(in_dim,out_dim,3,2,1),
			nn.LeakyReLU(0.2,inplace=True),
			nn.BatchNorm2d(out_dim)
			)
		return layer
	def discrminator(self,c):
		model=nn.Sequential(
			self.block(c,64),
			self.block(64,128),
			self.block(128,256),
			self.block(256,512)
			)
		return model

	def forward(self,img):
		validity=self.model(img)
		validity=validity.view(validity.size(0),-1)
		validity=self.clf_layer(validity)
		return validity