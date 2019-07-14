'''
Based on pytorch

Two layers test :
	1. softmax in fast-rcnn
	2. up-sampling in YOLO V3
'''

import torch
import torch.nn.functional as F
import numpy as np
import time, timeit

a=[0 for x in range(0,53*39)]

t1=time.time()
for i in a:
    b=np.arange(256).reshape(1,256) #change to row vector
    Input_of_Softmax = torch.Tensor(b)
    #https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/model/faster_rcnn.py         line: 257
    output_of_Softmax = F.softmax(Input_of_Softmax, dim=1)   #dim=1 means the row direction
print(output_of_Softmax)
t2=time.time()
T=t2-t1
print(T)

'''
https://blog.csdn.net/shenxiaolu1984/article/details/51036677#comments
from this post, we coulf find the output from softmax is (K+1)*p
where, K means K categories (maximum 1000); P means the probabilities to each category. 
https://zhuanlan.zhihu.com/p/24916624?refer=xiaoleimlnote
https://blog.csdn.net/sloanqin/article/details/51545125

Fast-RCNN
input: 224*224*3
softmax : 1*1*256     18 nodes  （但是这个也有可能是softmax前的fc层。因为上面的softmax函数根本就没有nodes的输入）
softmax : 18 numbers  output: 9*2  (one grid cell has 9 anchor boxes, one anchor box has 2 possibilities )
totally 51*39 grid cells. It means 51*39 times of softmax layer
'''
