#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-10-23 16:25:09
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from torch import nn
import torch
class Generator(nn.Module):
	def __init__(self,n_filters=32,kernel_size=5,in_channel=3,l=4):
		super(Generator,self).__init__()
		self.pre=nn.Sequential(
			nn.Conv2d(in_channel,n_filters,kernel_size,padding=(kernel_size-1)//2),
			nn.BatchNorm2d(n_filters),
			nn.LeakyReLU(),
			)##256
		self.down1=BlockDown(n_filters,n_filters,kernel_size,l)
		self.down2=BlockDown(n_filters,2*n_filters,kernel_size,l)
		self.down3=BlockDown(2*n_filters,4*n_filters,kernel_size,l)
		self.down4=BlockDown(4*n_filters,8*n_filters,kernel_size,l)

		self.up4=BlockUp(8*n_filters,4*n_filters,kernel_size,l)
		self.up3=BlockUp(8*n_filters,2*n_filters,kernel_size,l)
		self.up2=BlockUp(4*n_filters,1*n_filters,kernel_size,l)

		self.final_block=nn.Sequential(
			nn.Conv2d(2*n_filters,n_filters,1,padding=0),
			DenseBlock(k=n_filters,l=l,filter_size=kernel_size),
			nn.BatchNorm2d(n_filters),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2,mode="nearest"),
			nn.Conv2d(n_filters,in_channel,1,padding=0),
			nn.Tanh())

	def forward(self,x):
		x=self.pre(x)
		en1,pool1=self.down1(x)
		en2,pool2=self.down2(pool1)
		en3,pool3=self.down3(pool2)
		en4,pool4=self.down4(pool3)

		de4=self.up4(pool4)#256
		tmp_de4=torch.cat([de4,en4],dim=1)

		de3=self.up3(tmp_de4)
		tmp_de3=torch.cat([de3,en3],dim=1)

		de2=self.up2(tmp_de3)
		tmp_de2=torch.cat([de2,en2],dim=1)		

		de1=self.final_block(tmp_de2)
		return de1

class BlockUp(nn.Module):
	def __init__(self,in_dim,out_dim,kernel_size,l):
		super(BlockUp,self).__init__()
		self.block=nn.Sequential(
			nn.Conv2d(in_dim,out_dim,kernel_size,padding=(kernel_size-1)//2),
			DenseBlock(k=out_dim,l=l,filter_size=kernel_size),
			nn.BatchNorm2d(out_dim),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2,mode="nearest"))
	def forward(self,x):
		return self.block(x)
class BlockDown(nn.Module):
	def __init__(self,in_dim,out_dim,kernel_size,l):
		super(BlockDown,self).__init__()
		self.block=nn.Sequential(
			nn.Conv2d(in_dim,in_dim,kernel_size,padding=(kernel_size-1)//2),
			DenseBlock(k=in_dim,l=l,filter_size=kernel_size),
			nn.BatchNorm2d(in_dim),
			nn.LeakyReLU(),
			)##256
		self.block_down=nn.Sequential(
			nn.Conv2d(in_dim,out_dim,kernel_size,padding=(kernel_size-1)//2,stride=2),
			nn.BatchNorm2d(out_dim),
			nn.LeakyReLU()			
			)
	def forward(self,x):
		out=self.block(x)
		down=self.block_down(out)
		return out,down
class DenseBlock(nn.Module):
	"""docstring for DenseBlock"""
	def __init__(self,k,l,filter_size=5):
		super(DenseBlock, self).__init__()
		self.layers=nn.ModuleList()
		self.l=l
		for i in range(l):
			self.layers.append(
				nn.Sequential(
					nn.BatchNorm2d(k+i*k),
					nn.LeakyReLU(),
					nn.Conv2d(k+i*k,k,filter_size,padding=(filter_size-1)//2)
				)
			)

	def forward(self,x):
		for i in range(self.l):
			y=self.layers[i](x)
			x=torch.cat([y,x],dim=1)
		return y

if __name__=="__main__":
	pass
