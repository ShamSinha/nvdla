#!/usr/local/bin/python3
import sys 
import math
''' Given time taken to in each module of NVDLA along with the time between modules we can calculate the total time taken
	to compute a layer in a neural network 

	Parameters to keep in minds (there are too many of them! Good luck keeping track of all!!)
		1) padding 
		2) zero addition 
		3) input data cube 
		4) kernel data cube + number of kernels
		5) wire width 
		6) register size 
		7) memory mapping of data 
'''

''' The pipeline of the overall process is as follows :					-------->--------	
																		|				|
																		|				|
			DRAM --- > SRAM ---- > CBUF ---- > CMAC ---- > CACC-Assembly Group 	   CACC Delivery Group ---- > SDP ---
						|																							|	
						|																							|			
						------------------------------------------<--------------------------------------------------																							 


 First we define the pipes associated with the entire process this include 
 	Pipe 1 -- DRAM -- > SRAM -- > CBUF 
 	Pipe 2 -- CBUF -- > CMAC -- > Assembly Group
 	Pipe 3 -- Delivery Group -- > SDP -- > SRAM

					 L2_pipe
					  /	  \
					 /     \
					/       \
				   /	     \
				Pipe 1       L1_pipe
							   /\
							  /  \
							 /    \
							/      \
						  Pipe 2   Pipe 3


This hierarchy represents how the three pipes join together to form levels of pipeline ultimately leading to a top level pipeline that functions for part
duration of the pipeline process 

	Any layer in YOLOv3 has an input data cube , weight kernel data (+ total kernels) 
	
	size_layer_i  = HxWxC   where i is denoting layer = {1,2,3,4,5,....N}
	size_weight_i = RxSxCxK where i is denoting layer = {1,2,3,4,5,....N}

	The size of SRAM can be determined based on MAX(size_layer_i + size_weight_i)

	For YOLOv3 this values ~ 6.5 MB  
	Now to determine the total SRAM size we also need to calculate the additional buffer size required as the output would also be stored into the same SRAM

 '''

def size_SRAM():
 	''' Input two files 
 			Weight data file containing size of each weights in each layer
 			Input feature data file containing size of each feature in each layer
 		Output 
 			max_size of input + weight in any layer
 	'''
 	weights = []
 	feature = []
 	total = []
 	t = 0
 	with open('weight-size.txt', 'r') as f:
 		for line in f:
 			weights.append(line)
 	with open('input-size.txt', 'r') as f:
 		for line in f:
 			feature.append(line)
 	for i in range(len(weights)):
 		t = int(weights[i]) + int(feature[i])
 		total.append(t)
 	max_sum = max(total)
 	denominator = 1024*1024*8 	# to get answer in MB
 	max_sum = max_sum/denominator
 	del weights
 	del feature
 	del total
 	return (math.ceil(round(max_sum,3)))  # for the moment we take SRAM as 8 MB by adding 2.7 MB of additional storage space in on-chip SRAM

''' Lets us first try to achieve some answer for just one layer of a neural network 
	This would be followed by second layer and the third one which will encompass the Resnet Layer
	Once the three layers are done the same process will repeat continously until the entire network is completed
	Remember to commit after each layer completion 
'''








































def main():
	maxi = size_SRAM()


if __name__ == '__main__':
	main()