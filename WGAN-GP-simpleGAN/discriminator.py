#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-21 10:38:21
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch import nn

class Discriminator(nn.Module):
	def __init__(self,h,w,c):
		super(Discriminator,self).__init__()

		self.model=self.discrminator(h,w,c)

	def discrminator(self,h,w,c):
		model=nn.Sequential(
			nn.Linear(int(h*w*c),512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Linear(512,256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Linear(256,1),
			)
		return model

	def forward(self,img):
		img_flatten=img.view(img.size(0),-1)
		validity=self.model(img_flatten)
		return validity
