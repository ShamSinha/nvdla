import torch
import numpy as np
import timeit
import matplotlib.pyplot as plt
import statistics
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

	t1=timeit.timeit()    #start the time
	#the first upsample layer
	model=torch.nn.Upsample((26, 26), mode='bilinear', align_corners=True)
	model(input_1)
	#pre=model(input_1)
	#print(pre)
	#the second upsample layer
	model=torch.nn.Upsample((52, 52), mode='bilinear', align_corners=True)
	model(input_2)

	t2=timeit.timeit()    #stop the time
	T=t2-t1
	return T         #print the total time

if __name__ == '__main__':
	time_array = []
	j =0
	for i in range(10000):
		if(i%500 == 0):
			j += 1
			print("Computing " + "."*(j))
		time = upsampling()
		time_array.append(time)

	for val in time_array:
		if val < 0:
			time_array.remove(val)
			val = -1*val
			time_array.append(val)
	
	count = 0

	for val in time_array:
		if val<0:
			count += 1

	print(count)
	print("Median time: ", statistics.median(time_array))
	plt.plot(time_array)
	plt.xlabel("Iteration Number")
	plt.ylabel("Time")
	plt.show()
