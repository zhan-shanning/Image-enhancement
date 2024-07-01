import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		# b,3,h,w-> b,32,h,w
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
		# b,32,h,w-> b,32,h,w
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
		# b,32,h,w-> b,32,h,w
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
		# b,32,h,w-> b,32,h,w
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)

		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,3,3,1,1,bias=True)




		
	def forward(self, x):

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4(x3))

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

		x_r = torch.sigmoid(self.e_conv7(torch.cat([x1,x6],1)))



		# 改进的全局伽马校正

		enhance_image= x_r*torch.pow(x,0.25)+(1-x_r)*(1-torch.pow(1-x,0.25))

		return enhance_image,enhance_image,x_r

# 对模型参数量的相关测试
# model=enhance_net_nopool()
# total_params = sum(p.numel() for p in model.parameters())
# print("模型参数数量：", total_params)
#
#
# import torch
#
#
# iterations = 300   # 重复计算的轮次
#
# device = torch.device("cuda:0")
# model.to(device)
#
# random_input = torch.randn(1, 3, 640, 640).to(device)
# starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#
# # GPU预热
# for _ in range(50):
#     _ = model(random_input)
#
# # 测速
# times = torch.zeros(iterations)     # 存储每轮iteration的时间
# with torch.no_grad():
#     for iter in range(iterations):
#         starter.record()
#         _ = model(random_input)
#         ender.record()
#         # 同步GPU时间
#         torch.cuda.synchronize()
#         curr_time = starter.elapsed_time(ender) # 计算时间
#         times[iter] = curr_time
#         # print(curr_time)
#
# mean_time = times.mean().item()
# print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))









