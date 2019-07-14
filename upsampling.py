import torch
import numpy as np
import time, timeit

'''
two upsample layers
1. 85 layer
   input 13*13*256     ---> 2*  ---> output: 26*26*256
2. 97 layer
	input 26*26*128    ---> 2* ----> output: 52*52*128
3. 
'''
#https://pytorch.org/docs/stable/nn.html
#https://www.aiuai.cn/aifarm605.html
#https://www.cnblogs.com/xzcfightingup/p/7598293.html
def upsampling():
	#https://discuss.pytorch.org/t/transform-model-to-xxx-proto-failed/24353
	input_1=torch.rand(1, 256, 13, 13)
	#print("input", input_1)
	#print(input_1.size())
	input_2=torch.rand(1, 128, 26, 26)

	t1=time.time()    #start the time
	#the first upsample layer
	model=torch.nn.Upsample((26, 26), mode='bilinear', align_corners=True)
	model(input_1)
	#pre=model(input_1)
	#print(pre)
	#the second upsample layer
	model=torch.nn.Upsample((52, 52), mode='bilinear', align_corners=True)
	model(input_2)

	t2=time.time()    #stop the time
	T=t2-t1
	return T         #print the total time

if __name__ == '__main__':
	
	time = upsampling()
	print(time)