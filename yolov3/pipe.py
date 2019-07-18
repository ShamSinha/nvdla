#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

# Ownership
# Improve this section later
# __author__ = "Aditya Prasad, Shubham Kumar, Chen Haoji"
# __copyright__ = “Copyright 2019, AI ACCELERATOR PROJECT”
# __credits__ = ['Aditya Prasad','Shubham Kumar','Chen Haoji', 'Moonki', 'Yangyi', 'Dhyeo']
# __license__ = “Decide Later”
# __version__ = “0.1.0”
# __maintainer__ = “Digital Systems Lab, Hanyang University”
# __email__ = “nvdlagroup@gmail.com”
# __status__ = “Dev”

''' YOLOv3 INFERENCE CALCULATION'''

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
 	Pipe 0 -- DRAM -- > SRAM
 	Pipe 1 -- SRAM -- > CBUF 
 	Pipe 2 -- CBUF -- > CMAC -- > Assembly Group
 	Pipe 3 -- Delivery Group -- > SDP -- > SRAM
 	Pipe 4 -- Assembly Group -- > Delivery Group
 	Copy   -- SRAM -- > DRAM

				 	L2_pipe
					 /	  \
					/      \
				   /        \
				  /	         \
			L1_1_Pipe    L1_2_pipe
			 /	|	           /\
		    /	|		      /  \
		   /	|		     /    \
		  /		|	        /      \
	 Pipe 1	  Pipe 4      Pipe 2   Pipe 3


This hierarchy represents how the three pipes join together to form levels of pipeline ultimately leading to a top level pipeline that functions for part
duration of the pipeline process 

	Any layer in YOLOv3 has an input data cube , weight kernel data (+ total kernels) 
	
	size_layer_i  = HxWxC   where i is denoting layer = {1,2,3,4,5,....N}
	size_weight_i = RxSxCxK where i is denoting layer = {1,2,3,4,5,....N}

	The size of SRAM can be determined based on MAX(size_layer_i + size_weight_i)

	For YOLOv3 this values ~ 5.5 MB  
	Now to determine the total SRAM size we also need to calculate the additional buffer size required as the output would also be stored into the same SRAM

In the code following this introduction Functions are labelled as :
	Operation: if time is being calculated there
	Non-Operation : if time is not being calculated there these would include condition checker, other calculators etc.

In each operation function the state of the system is defined in terms of active or idle 
This is pretty important as it will tell us what modules are functional at any time 

Important Note - time in this file means cycles. 	

The wait time in cases where multiple read/writes must happend with a memory unit, can be modelled as a stochastic varible. The specifics can be hammered later

The goal is that the program outputs an equation for computation of an image input. This can be done only when all the layers in the network have been computed. 


Until now the basic calculation model for first layer has been programmed (hopefully it works and is correct). The chunks that we have divided the layer into consitute the smallest unit of the layer computed at a time 
Now I will determine how the pipeline works between these chunks. So far the pipeline for one chunk was examined. As we start computing more chunks in parallel we will form a the higher level pipeline as mentioned in the diagram above

Important Note: It is assumed that the next layer of network wouldn't start until the previous one is totally complete. This is how the architecture was built

On the discussion of Multiple Networks and their inference time computation : 
	So far I have designed the calculation time for just one chunk of data of one layer. This chunk of data is fundamental to any network with Convolution operation as the key operation 
	Don't know if it generalizes well for fc layers (maybe it does ???)

	So for any network, the design of the code should be able to handle Convolution operation pretty well. 			

Changes ======================
1) DRAM is single port and SRAM is dual port so make changes if necessary	
2) SRAM_size was decided arbitrarily ( the additional 2MB was arbitrary)
3) chunk_size_DRAM_SRAM was picked arbitrarily ( we can have better logic for this) ( maybe its not much of a problem anyway. Confirm)

1) CBUF_
'''

import sys 
import math
import logging 
import traceback
import timeit

from upsampling import upsampling
from softmax import softmax
# lists holding the various sizes in bits
weights = []
feature = []
total = []


# logging 
logging.basicConfig(filename="./pipe.log", 
                    format='%(asctime)s %(message)s',filemode='w') 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)																	   #7																												#18

t0_GDDR6_512bits,t1_GDDR6_512bits,gddr6_clk,t0_SRAM_512bits,t1_SRAM_512bits,sram_clk,t0_BDMA_20Deep,t0_CDMA,Writing_bits,t0_CBUF,t1_CBUF,t0_CSC,t0_CMAC,t0_CACC_Adder,t0_Assembly,t1_Assembly,t0_Delivery,t1_Delivery,t0_truncation,Assembly_writing_bits,t0_SDP,t1_SDP,delay_dram_sdp,data_dram_sdp_bits = (3,2.4,1500000000,2,2,1000000000,0,4,512,0,16,7,7,0,0,1,1,1,10,544,1,1,0,512)  # correct values used now for debugging

''' Important Note - input-size.txt =  75 lines
					 weight-size.txt = 75 lines '''

'''Non-Operation'''
def size_SRAM():
 	''' Input two files 
 			Weight data file containing size of each weights in each layer
 			Input feature data file containing size of each feature in each layer
 		Output 
 			max_size of input + weight in any layer
 	'''
 	global weights
 	global feature
 	global total
 	t = 0
 	with open('weight-size.txt', 'r') as f:
 		for line in f:
 			weights.append(int(line))
 	with open('input-size.txt', 'r') as f:
 		for line in f:
 			feature.append(int(line))
 	for i in range(len(weights)):
 		t = int(weights[i]) + int(feature[i])
 		total.append(t)
 	max_sum = max(total)
 	denominator = 1024*1024*8 	# to get answer in MB
 	max_sum = max_sum/denominator
 
 	return (math.ceil(round(max_sum,3)))  # for the moment we take SRAM as 8 MB by adding 2.7 MB of additional storage space in on-chip SRAM

# check that input and weight arrays are equal in length. This is not working properly 
if(len(weights) == len(feature)):
	logging.info("Continuing.. ")
else:
	logging.info("PROBLEM.... Exiting for now")
	sys.exit(0)
''' Lets us first try to achieve some answer for just one layer of a neural network 
	This would be followed by second layer and the third one which will encompass the Resnet Layer
	Once the three layers are done the same process will repeat continously until the entire network is completed
	Remember to commit after each layer completion 
'''

GDDR6_writing_time = t0_GDDR6_512bits  # time taken to write 512 bits of data to GDDR6
GDDR6_reading_time = t1_GDDR6_512bits  # time taken to read 512 bits of data from GDDR6
GDDR6_clock_freq = gddr6_clk

SRAM_size = size_SRAM()  #MB  where did the 2 come from we need to figure this out
SRAM_size_in_bits = SRAM_size*1024*1024*8 # sram size in bits 
chunk_size_DRAM_SRAM = 0.5*1024*1024*8      # = SRAM_size_in_bits - previous_output_in_SRAM or total_data_to_transfer (<SRAM_size_in_bits)
SRAM_writing_time = t0_SRAM_512bits    # time taken to write 512 bits of data to SRAM
SRAM_reading_time = t1_SRAM_512bits    # time taken to write 512 bits of data to SRAM
SRAM_clock_freq = sram_clk

delay_bdma = t0_BDMA_20Deep			   # time spent in bdma (useful for first transfer) difference between how long it takes for the first entry of sequence from GDDR6 to cross BDMA

SRAM_CHUNKS_FROM_DRAM = []  # stores the values for each chunk that will be transferred from DRAM --> SRAM

previous_output_in_SRAM = 0 # initially previous output is zero

'''Operation'''
def time_DRAM_SRAM(direction, index_input,index_sram_chunk):
	''' check for the direction in which the data should be transferred 
		right means DRAM -> SRAM
		left  means SRAM -> DRAM
	
	Note : DRAM  ------ active 
		   SRAM  ------ active
		   Other ------ idle   
	'''
	print("///IN time_DRAM_SRAM()////")
	dummy_time = 0 # garbage time (throw away later)
	global GDDR6_reading_time, weights, feature, SRAM_writing_time, delay_bdma, SRAM_CHUNKS_FROM_DRAM,GDDR6_clock_freq,SRAM_clock_freq
	stages = {'dram':'active', 'sram':'active', 'cbuf':'idle', 'cmac':'idle', 'Assembly':'idle', 'Delivery':'idle', 'SDP':'idle'}
	logger.info("Active and Idle stages : {}".format(stages))
	
	total_data_to_transfer = feature[index_input] + weights[index_input] # total data to be transferred from DRAM -> SRAM 

	if direction == 'right':
		logging.info("DRAM -> SRAM")
		
		total_transfer_time = 0
		
		chunks_read_size = 512
		if (can_sram_fit(total_data_to_transfer)):
			''' At the end of this all data for convolution is going to present in SRAM, now DRAM is idle'''

			logger.info("SRAM can fit all data... Proceeding...")
			total_chunks_to_read = math.ceil(SRAM_CHUNKS_FROM_DRAM[index_sram_chunk]/chunks_read_size)
			transfer_time_first_chunk = GDDR6_reading_time + delay_bdma + SRAM_writing_time  # time taken to transfer just the first chunk of 512 bits 

			# for the remaining chunks
			for chunk in range(total_chunks_to_read-1):
				# for the remaining chunks, since Pipe 1 is established now, the time taken now would be only as fast as the slowest component i.e. GDDR6 read time
				# We could have multiplied, but for loop gives a better feel for transfer I think!
				total_transfer_time += GDDR6_reading_time

			# Now the total time taken can simply be added for first set of data to transfer into SRAM 
			total_transfer_time = total_transfer_time + transfer_time_first_chunk
			logging.info("Total time to transfer : {}".format(total_transfer_time))
			logging.info("At this point the data for Layer " + str(index_input)+ " of YOLOv3 has been transferred to SRAM")
			
		else:
			''' since the whole data cannot be transferred to SRAM we need to split the data and transfer smaller amounts at a time 
				refer to the pseudo code to figure this out as there are certain other things that need to be taken care of before we decide how to and how much
				to split 
				Note that the core logic for calculating time would remain the same '''
			logger.info("SRAM cannot fit all data at once... Proceeding with smaller chunks...")
			total_chunks_to_read = int(SRAM_CHUNKS_FROM_DRAM[index_sram_chunk]//chunks_read_size) + int(SRAM_CHUNKS_FROM_DRAM[index_sram_chunk]%chunks_read_size)
			transfer_time_first_chunk = GDDR6_reading_time + delay_bdma + SRAM_writing_time # time taken to transfer just the first chunk of 512 bits 

			# for the remaining chunks
			for chunk in range(total_chunks_to_read-1):
				# for the remaining chunks, since Pipe 1 is established now, the time taken now would be only as fast as the slowest component i.e. GDDR6 read time
				# We could have multiplied, but for loop gives a better feel for transfer I think!
				total_transfer_time += GDDR6_reading_time
			# Now the total time taken can simply be added for first set of data to transfer into SRAM 
			total_transfer_time = total_transfer_time + transfer_time_first_chunk
			logging.info("Total time to transfer : {}".format(total_transfer_time))
			logging.info("At this point the data for Layer " + str(index_input)+ " of YOLOv3 has been transferred to SRAM")		
		
	elif direction == 'left':
		''' write to code for SRAM to DRAM transfer and get the total time in this case '''
		logger.info("Exiting... Invalid option")
		sys.exit(0)

	return total_transfer_time

'''Non-Operaton'''
def length_SRAM_CHUNKS_FROM_DRAM(index_input):
	''' return length of SRAM_CHUNKS_FROM_DRAM for every new layer '''
	if(index_input == 0):
		total_data_to_transfer = feature[index_input] + weights[index_input]
	else:
		total_data_to_transfer =  weights[index_input]
	total_chunks = generate_chunks_for_sram(total_data_to_transfer)
	make_chunks_SRAM_CBUF()
	#print("CBUF_CHUNKS_FROM_SRAM: {}".format(CBUF_CHUNKS_FROM_SRAM))
	return total_chunks


'''Non-Operation'''
def generate_chunks_for_sram(size):
	''' generate chunks and prepare SRAM_CHUNKS_FROM_DRAM '''
	
	global SRAM_CHUNKS_FROM_DRAM,previous_output_in_SRAM,SRAM_size_in_bits


	remaining_SRAM_space = SRAM_size_in_bits - previous_output_in_SRAM
	print("----------", remaining_SRAM_space, " ", previous_output_in_SRAM)
	if(remaining_SRAM_space <= SRAM_size_in_bits and remaining_SRAM_space >0): 
		
		# if(size < remaining_SRAM_space):
		# 	total_chunks = 1
		# 	chunk_size = size
		# else:
		# 	total_chunks = math.ceil(size/chunk_size_DRAM_SRAM)
		# 	chunk_size = chunk_size_DRAM_SRAM #remaining_SRAM_space
		total_chunks = math.ceil(size/chunk_size_DRAM_SRAM)
		chunk_size = chunk_size_DRAM_SRAM #remaining_SRAM_space
		for i in range(total_chunks):
			SRAM_CHUNKS_FROM_DRAM.append(chunk_size)
		
	else:
		logging.info("No transfer. Chunk size (chunk_size_DRAM_SRAM) too big....")
		sys.exit(0)
	return total_chunks

''' Non-Operation '''
def can_sram_fit(total_size):
	'''this function is meant to check if the data we plan on sending to SRAM can be accomodated by SRAM 
		if yes:
			continue with transfer to sram 
	   		return True 
		if no :
			split the data now into smaller chunks and transfer small chunks that can be transferred 
	'''
	
	denominator = 1024*1024*8 	# to get answer in MB
	total_size = total_size / denominator
	global SRAM_size, SRAM_CHUNKS_FROM_DRAM
	if(SRAM_size >= total_size):
		logger.info("SRAM_CHUNKS_FROM_DRAM holds the total_size of feature + weights : {} MB".format(total_size))
		SRAM_CHUNKS_FROM_DRAM.append(total_size*denominator)
		return True
	return False

''' Now we define the logic for transfer of data from SRAM to CBUF, 
	
	SRAM --> CDMA --> CBUF

	Since CBUF is smaller than SRAM we must be careful how much data is transferred at a time.
	Once the requisite amount of data has been transferred we can move forward with computation 
	This process of transfer of data from SRAM --> CBUF is going to happen multiple times 
	So first we calculate for the case when the first chunk is to be transferred '''

CBUF_CHUNKS_FROM_SRAM = []  # stores the values for each chunk that will be transferred from SRAM --> CBUF 
							  # the index of CBUF_CHUNKS_FROM_SRAM would give us the SRAM_CHUNK that we are currently looking at 
CBUF_SIZE = 524288*8 # size of CBUF in bits 
CBUF_CHUNK_SIZE = int(CBUF_SIZE/16)
''' Important Note - Everytime we process a layer the values in arrays CBUF_CHUNKS_FROM_SRAM and SRAM_CHUNKS_FROM_DRAM need to be updated again. i.e. the arrays must be 
	emptied and refilled for the new layer. However while working on one layer, no need to change'''

'''Non-Operation'''
def make_chunks_SRAM_CBUF():
	''' create the appropriate chunks that will be transferred to CBUF for computation
		store the sizes in CBUF_CHUNKS_FROM_SRAM, this will be used later 
		SRAM_SUB_I --> CBUF_CHUNKS_FROM_SRAM where I are all the chunks we must transfer from SRAM to CBUF over time 
		Compute SRAM_SUB_I sizes. 
	'''

	global CBUF_CHUNKS_FROM_SRAM, weights, feature, SRAM_CHUNKS_FROM_DRAM, CBUF_SIZE, CBUF_CHUNK_SIZE	
	for count,sram_chunk in enumerate(SRAM_CHUNKS_FROM_DRAM):
		''' for each sram_chunk in SRAM_CHUNKS_FROM_DRAM we need it split it further so that CBUF is filled all the time 
			except maybe in the last case when only remainder in SRAM would be transferred '''
		if(CBUF_CHUNK_SIZE <= CBUF_SIZE):
			cbuf_chunk_size = CBUF_CHUNK_SIZE      # size of each cbuf_chunk that will be stored in cbuf_chunk_array that makes up one entry in CBUF_CHUNKS_FROM_SRAM
			remainder = sram_chunk%CBUF_CHUNK_SIZE # this is the last entry in cbuf_chunk_array 
			total_cbuf_chunks = math.ceil(sram_chunk/CBUF_CHUNK_SIZE )
			cbuf_chunk_array = []  			 # this list will store all the cbuf_chunk_sizes and this will be appended to CBUF_CHUNKS_FROM_SRAM
			
			for i in range(total_cbuf_chunks):
				if (i == total_cbuf_chunks-1 and remainder !=0 ):
					cbuf_chunk_array.append(remainder)

				cbuf_chunk_array.append(cbuf_chunk_size)

			CBUF_CHUNKS_FROM_SRAM.append(cbuf_chunk_array)   # CBUF_CHUNKS_FROM_SRAM will be created when all the indices in SRAM_CHUNKS_FROM_DRAM have been parsed
															 # Now, each individual values of CBUF_CHUNKS_FROM_SRAM will feed into the Pipe 2 
		else:
			logging.info("To Big for CBUF... Exiting for now")
			sys.exit(0)


def length_CBUF_CHUNKS_FROM_SRAM():
	''' return length of CBUF_CHUNKS_FROM_SRAM where every entry is a list '''
	
	global CBUF_CHUNKS_FROM_SRAM
	
	CBUF_CHUNKS_FROM_SRAM_entries = []
	for i in range(len(CBUF_CHUNKS_FROM_SRAM)):
		length = len(CBUF_CHUNKS_FROM_SRAM[i])
		CBUF_CHUNKS_FROM_SRAM_entries.append(length)
	return CBUF_CHUNKS_FROM_SRAM_entries


delay_cdma = t0_CDMA   # time spent for the first set of values transferred from SRAM to CDMA. Same idea as BDMA. More accuracy can be achieved later
SRAM_writing_bandwidth = Writing_bits   
CBUF_reading_bandwidth = 192*8 # bits to CBUF
CBUF_Reading_time = t0_CBUF
CBUF_Writing_time = t1_CBUF
CBUF_writing_bandwidth = 256*8 # bits to CSC


'''Operation'''
def time_SRAM_CBUF(index_sram_chunk, index_cbuf_chunk):
	''' calculate the total time taken to transfer one sram_chunk into CBUF 
		The sram_chunks stored in SRAM_CHUNKS_FROM_DRAM would be transferred into CBUF one at a time. Not adjacently. First one will be transferred to CBUF
		The time taken to do so would be calculated and then the next one would be calculated when the first one has been computed completely in the remaining pipeline
		Note : SRAM  ------ active 
		   	   CBUF  ------ active
		   	   Other ------ idle  
		 '''
	
	stages = {'dram':'idle', 'sram':'active', 'cbuf':'active', 'cmac':'idle', 'Assembly':'idle', 'Delivery':'idle', 'SDP':'idle'}
	logger.info("Active and Idle stages : {}".format(stages))
	
	chunk_to_be_read = CBUF_CHUNKS_FROM_SRAM[index_sram_chunk][index_cbuf_chunk]  # bits 
	total_chunks_to_read = math.ceil(chunk_to_be_read/SRAM_writing_bandwidth )  # total number of chunks we need to read from SRAM
	total_transfer_time = 0
	first_transfer_time = SRAM_writing_time + delay_cdma + CBUF_Reading_time
	logging.info("SRAM --> CBUF")
	
	for chunk in range(total_chunks_to_read-1): # for the remainder of chunks we just add the time taken to read from SRAM
		total_transfer_time += SRAM_writing_time

	total_transfer_time = total_transfer_time + first_transfer_time
	logging.info("Total time to transfer : {}".format(total_transfer_time))
	logging.info("At this point chunk number {} stored in CBUF_CHUNKS_FROM_SRAM has been transferred to CBUF. This fills CBUF".format(index_cbuf_chunk))

	return total_transfer_time
	''' At the end of this, the first chunk that needs to computed is available in CBUF. '''

''' Now that CBUF is full we can start the next stage. This will activate Pipe 2.
	We will transfer data worth one atomic operation at a time. 

	CBUF --> CSC --> CMAC --> Assembly group 

	The first set of data will fill the pipeline first and subsequently, Pipe 2 would be activated. We will continue this process until the entire 
	CBUF is emptied and the values are stored in Assembly group. 
		
		Important Note: At this stage we consider that Assembly group has enough size to store the results generated from the input data present in CBUF
	Once the CBUF is emptied, Data from Assembly group will be transferred into Delivery Group 

'''

size_atomic_op = 16896 #41472, 25088, 16896, , #bits is the total size of input-- 1x1x64 + kernel--1x1x64x16 while considering int8 precision.
delay_csc = t0_CSC
delay_cmac = t0_CMAC
delay_adder_array = t0_CACC_Adder

Assembly_reading_time = t0_Assembly
Assembly_writing_time = t1_Assembly
Delivery_reading_time = t0_Delivery
Delivery_writing_time = t1_Delivery

'''Operation'''
def time_CBUF_Assembly(index_sram_chunk, index_cbuf_chunk):
	''' calculate the time taken to transfer data from CBUF to Assembly group 
		Note : CBUF  		  ------ active 
		   	   CSC  		  ------ active
		   	   CMAC 		  ------ active
		   	   Assembly_Group ------ active 
		   	   Other          ------ idle 

		   	Important Note that Other ------ idle might not stay true when the L1_Pipe is established. '''
	
	stages = {'dram':'idle', 'sram':'idle', 'cbuf':'active', 'cmac':'active', 'Assembly':'active', 'Delivery':'idle', 'SDP':'idle'}
	logger.info("Active and Idle stages : {}".format(stages))

	global size_atomic_op, CBUF_CHUNKS_FROM_SRAM,delay_csc, delay_cmac ,delay_adder_array,CBUF_Writing_time,CBUF_writing_bandwidth
	#print(index_sram_chunk, " ", index_cbuf_chunk)

	chunk_in_cbuf = CBUF_CHUNKS_FROM_SRAM[index_sram_chunk][index_cbuf_chunk]
	total_atomic_reads_from_cbuf = math.ceil(chunk_in_cbuf/size_atomic_op)
	
	# Since the bandwidth of CBUF is limited to CBUF_writing_bandwidth = 256*8, and the required size for computation is per atomic_operations is size_atomic_op
	# to send this data to CMAC would require multiple reads from CBUF 
	# Therefore
	total_reads_from_cbuf_for_one_atomic_op = math.ceil(size_atomic_op/CBUF_writing_bandwidth) # total reads from CBUF for one atomic operation data.
	time_to_read_first_atomic_chunk_into_CMAC = CBUF_Writing_time + delay_csc + delay_cmac + delay_adder_array  # this is the first chunk of atomic data that can be read from CBUF due to its limited bandwidth
	time_to_read_one_atomic_op = 0
	logging.info("CBUF --> CSC --> CMAC --> Assembly_Group")

	for i in range(total_reads_from_cbuf_for_one_atomic_op -1):
		'''for the remaing chunks for this atomic operation, since the pipeline is setup we can simply add the time to read from CBUF '''
		time_to_read_one_atomic_op += CBUF_Writing_time     # time taken to read one atomic operation equivalent of data into CMAC
	
	time_to_read_one_atomic_op = time_to_read_one_atomic_op + time_to_read_first_atomic_chunk_into_CMAC

	''' At the end of this one set of atomic operations would be complete and result (partial sum) would be stored back in Assembly Group
		Now we must continue this process to empty the CBUF and store all the results in CACC_Assembly_Group 
		As the pipeline Pipe 2 is already setup after first atomic operation we can find the time taken by adding the total number of atomic operations that need to be computed
		For each atomic operation we will add the  time_to_read_one_atomic_op time to the total time '''

	total_transfer_time = 0 

	for i in range(total_atomic_reads_from_cbuf-1):
		'''since we have already computed the first atomic read'''
		total_transfer_time += time_to_read_one_atomic_op

	logging.info("Total time to transfer : {}".format(total_transfer_time))
	logging.info("At this point chunk number {} stored in CBUF_CHUNKS_FROM_SRAM has been computed and stored in Assembly_Group. This empties CBUF".format(index_cbuf_chunk))
	
	return total_transfer_time


delay_truncation = t0_truncation
size_partial_sum_generate_per_atomic_op = 544 # bits in case of int8
Assembly_writing_bandwidth = Assembly_writing_bits 
Assemby_data_size = 0
''' Now that the result of the first chunk that was transferred to CBUF has been computed and result stored in CACC_Assembly_Group, we can begin to unload 
	this data into Delivery Group. Total time taken to complete this procedure will be added to the inference time '''

'''Operation'''
def time_Assembly_Delivery(index_sram_chunk,index_cbuf_chunk):
	''' Data stored in Assembly is int34 for int8 data values, While data stored in Delivery is int32

		The result of 1 1x1x64 o 1x1x64 is 1 partial sum = int8
		Now this partial sum is extended and added with int34 to give 1 int34 value 
		Therefore for 1 atomic operation the total size of partial sums generated = 16x34 = 544 bits
	
		Now total number of atomic operations for the given chunk in CBUF is = total_atomic_ops
		Therefore size of Assembly_Group result data = total_atomic_ops * 544

		SO we have the total Assembly Group size. This will be truncated to int32 by truncation array and finally stored into Delivery Group
	
		Note : Assembly Group ---- active
			   Delivery Group ---- active
			   Other          ---- idle 
		While data is being transferred from Assembly to Delivery Group other operations would not be functional atleast for Pipe 2 and Pipe 3
		'''
	
	global Assembly_writing_bandwidth, delay_truncation, size_partial_sum_generate_per_atomic_op,CBUF_CHUNKS_FROM_SRAM,size_atomic_op, Assemby_data_size
	total_transfer_time = 0
	# calculate the total size in Assembly from the first chunk in CBUF 
	chunk_in_cbuf = CBUF_CHUNKS_FROM_SRAM[index_sram_chunk][index_cbuf_chunk]
	total_atomic_reads_from_cbuf = math.ceil(chunk_in_cbuf/size_atomic_op)
	Assemby_data_size = total_atomic_reads_from_cbuf*size_partial_sum_generate_per_atomic_op
	logging.info("Size of Assembly_Group {}".format(Assemby_data_size))
	logging.info("Assembly_Group ---> Delivery_Group")

	total_reads_from_assembly = math.ceil(Assemby_data_size/Assembly_writing_bandwidth)
	time_first_read_from_assembly = Assembly_writing_time + delay_truncation + Delivery_reading_time

	for i in range(total_reads_from_assembly -1):
		total_transfer_time += Assembly_writing_time

	total_transfer_time = total_transfer_time + time_first_read_from_assembly  

	logging.info("Total time to transfer : {}".format(total_transfer_time))
	logging.info("At this point chunk number: {} stored in CBUF_CHUNKS_FROM_SRAM has been computed and stored in Delivery Group. This empties Assembly".format(index_cbuf_chunk))
	
	return total_transfer_time

''' Now that the Delivery Group is filled up we can start sending the to SDP for post- processing. Subsequenlty this data would be put back in on-chip SRAM

Important Note: Since the Assembly is free, the Pipe 2 can also function concurrently with Pipe 3, here we have the option to pick which one is taking longer and start forming 
the L1_Pipe. '''

delay_sdp = t0_SDP
delay_sdp_resnet = t1_SDP
Delivery_Group_bandwidth = 512 # bits
data_size_Delivery_Group = 0   # data size in bits (inside delivery group)
precision_Delivery_SRAM = 32 # bits for all precisions 
precision_Delivery_SRAM = 34 # bits for int8
delay_DRAM_SDP = delay_dram_sdp # time it takes for data to be transferred from DRAM to SDP buffer
data_DRAM_SDP = data_dram_sdp_bits # number of 512bits data transferred from DRAM to SDP for the purpose resnet layer
SDP_internal_buffer_size = 32768 # RDMA size 4KB

'''Operation'''
def time_Delivery_SRAM(resnet_flag, cached_for_resnet):
	''' calculate the time taken to transfer all data in Delivery SRAM to on-chip SRAM
		Before that we calculate the size of data in Delivery Group. 
		Since data stored in Assembly Group is int34 for int8 and int32 in Delivery 
		We can calculate the size of data in Delivery Group through simple unitary method

		takes resent_flag as input to determine if resnet operation needs to be computed. 
	'''
	'''if resnet operation has to be performed we will wait for that to have to switch to delay_sdp_resnet 
	   Also at this point the DRAM would be transferring layer-2 values from MCIF to SDP buffer. This means that a separate pipeline would 
	   be established which would require to transfer from DRAM -->SRAM and we will have to compute the time it takes for this operation to happen 
	   Is it going to be parallel to Pipe1 + Delivery --> SDP 
	   Also be careful that we would also be writing back to DRAM the copy of current layer being computed so there could a wait or lag associated with this. 
	   MCIF can only write to or read from GDDR6 at a time. This means that there could be a wait period associated with resnet layers. Just think about this a little more
	'''
	''' For the resnet layer, we have to transfer the additional output of convolution operation from 2 layers back, this means when the first chunk is transferred from DRAM to SRAM
		we can use the time taken for than chunk to go from SRAM -> CBUF -> CMAC -> Assembly -> Delivery -> SDP, to store some(How much?) data into SDP buffer
		However for subsequent chunks we need to add the extra time it takes to fetch this data and move to SDP as we are also writing the results of the current SDP output 
		back into DRAM. But at the same time we are trying to fetch the output of convolution layer that happened before from DRAM. Both operation cannot proceed simultaneously'''
		# CHECK ONCE LATER
	
	global Assemby_data_size,Delivery_Group_bandwidth,Delivery_writing_time,GDDR6_reading_time,data_DRAM_SDP,SDP_internal_buffer_size

	data_size_Delivery_Group = (Assemby_data_size*precision_Delivery_SRAM*8)/34

	total_reads_from_delivery = math.ceil(data_size_Delivery_Group/Delivery_Group_bandwidth)
	logging.info("Size of Delivery_Group {}".format(data_size_Delivery_Group))
	logging.info("Delivery_Group ---> SDP ---> SRAM")
	if(not select_resnet(resnet_flag)):
		logging.info("Non Resnet Layer.. Proceeding..")
		time_first_read_from_delivery = Delivery_writing_time + delay_sdp + SRAM_reading_time
		total_transfer_time = 0

		for i in range(total_reads_from_delivery-1):
			total_transfer_time += Delivery_writing_time

		total_transfer_time = total_transfer_time + time_first_read_from_delivery

		logging.info("Total time to transfer : {}".format(total_transfer_time))
		logging.info("At this point first chunk of data stored in CBUF_CHUNKS_FROM_SRAM has been computed and we are now transferring this from Delivery Group back to SRAM. This empties Delivery Group")

	else:
		logging.info("Resnet Layer.. Proceeding..")
		
		RDMA_reads_from_DRAM = SDP_internal_buffer_size/512  # reads to fill RMDA buffer in SDP
		rdma_time_to_fetch_data_dram_sdp = GDDR6_reading_time*RDMA_reads_from_DRAM   # time to fetch 4KB data into SDP 
		total_reads_from_dram = cached_for_resnet/SDP_internal_buffer_size		# need cached output from 2 layers back
		time_first_read_from_delivery = Delivery_writing_time + delay_sdp_resnet + SRAM_reading_time
		total_transfer_time = 0

		for i in range(total_reads_from_delivery-1):
			if(Delivery_writing_time < rdma_time_to_fetch_data_dram_sdp):
				total_transfer_time += rdma_time_to_fetch_data_dram_sdp
			else:
				total_transfer_time += Delivery_writing_time
		total_transfer_time = total_transfer_time + time_first_read_from_delivery

		logging.info("Total time to transfer : {}".format(total_transfer_time))
		logging.info("At this point first chunk of data stored in CBUF_CHUNKS_FROM_SRAM has been computed and we are now transferring this from Delivery Group back to SRAM. This empties Delivery Group")

	return total_transfer_time, data_size_Delivery_Group

'''Non-Operation'''
def select_resnet(flag):
	if (flag == 1):
		return True
	return False

precision_of_output_in_SRAM = 8 # bits 
SRAM_reading_bandwidth = 512 # bits

'''Operation'''
def time_SRAM_DRAM():
	''' calculate time taken to transfer the output stored in SRAM coming from SDP. '''
	
	global GDDR6_writing_time, SRAM_reading_time, data_size_Delivery_Group, precision_of_output_in_SRAM,SRAM_reading_bandwidth

	Output_data_size_SRAM = (data_size_Delivery_Group*precision_of_output_in_SRAM)/precision_Delivery_SRAM
	total_reads_from_SRAM = math.ceil(Output_data_size_SRAM/SRAM_reading_bandwidth)
	time_first_read_from_SRAM = SRAM_reading_time + delay_bdma + GDDR6_writing_time
	total_transfer_time = 0
	logging.info("Size of Output data in SRAM {}".format(data_size_Delivery_Group))
	logger.info("SRAM -> DRAM")

	for i in range(total_reads_from_SRAM -1):
		total_transfer_time += SRAM_reading_time

	total_transfer_time = total_transfer_time + time_first_read_from_SRAM	
	logging.info("Total time to transfer : {}".format(total_transfer_time))
	logging.info("Copy of Output data from chunk in SRAM moved to DRAM")
	
	return total_transfer_time

Clock_freq = 2700000000   # 2.7 GHz

''' Now various chunks of data would be computed in parallel as the pipeline builds up '''

'''Non-Operation'''
def level_two_pipeline_cbuf_delivery(resnet_flag,index_sram_chunk,index_cbuf_chunk,cached_for_resnet):  # cbuf means following_chunk starting point and delivery means current_chunk following time 
	''' given that the first chunk of data has already started moving from Delivery Group back to SRAM we can begin the transfer the next chunk into CBUF
		At this stage two operations are being performed 
			1) reading from SRAM --> CBUF
			2) writing to SDP --> SRAM 
		Since SRAMIF can only handle one thing at a time we need to, the other operation will have to wait. 
	
	This one connects Pipe 1 and Pipe 2
	we assume that two chunks are available now at this point 
	current_chunk is in Pipe 2 
	following_chunk is in Pipe 1
	current_chunk is moving into SRAM as time progresses
	following_chunk is moving into Assembly group as time progresses starting at SRAM (but while the current_chunk was being transferred from Assembly to Delivery
	the following_chunk was being transferred from SRAM to CBUF, This is a seprate pipeline that we must consider. here too we pick the larger time and that would dependent on size of data)
	
	Also, for this we consider that following_chunk is already available in CBUF and current_chunk is available in Delivery_SRAM
	Since this process repeates till CBUF empties and also that the process is parallel, we need to pick the slowest one between Pipe 1 and Pipe 2
	
	'''
	pipe_picked = ["Delivery to SRAM", "CBUF to Assembly"]
	current_chunk_time, output_CBUF_chunk = time_Delivery_SRAM(resnet_flag,cached_for_resnet)
	following_chunk_time = time_CBUF_Assembly(index_sram_chunk,index_cbuf_chunk)
	max_time, pipe = MAX(current_chunk_time, following_chunk_time, pipe_picked)
	logging.info("Max time picked was: {0} which corresponds to {1}".format(max_time, pipe))
	return max_time, output_CBUF_chunk

''' Important Note : for the first chunk these two functions or the higher level pipeline makes no sense 
	Only when we start the with the next chunk this becomes apparent '''

'''Non-Operation'''
def level_two_pipeline_sram_assembly(index_sram_chunk,index_cbuf_chunk):
	''' this is very similar to the previous function only difference is that here we assume that 
		following_chunk is available in sram and has to be moved to cbuf 
		current_chunk is available in assembly group and has to be moved into delivery group

		The rest of the logic remains the same 
	'''
	pipe_picked = ["Assembly to Delivery", "SRAM to CBUF"]
	current_chunk_time = time_Assembly_Delivery(index_sram_chunk,index_cbuf_chunk)
	following_chunk_time = time_SRAM_CBUF(index_sram_chunk,index_cbuf_chunk)
	max_time,pipe = MAX(current_chunk_time, following_chunk_time,pipe_picked)
	logging.info("Max time picked was: {0} which corresponds to {1}".format(max_time, pipe))
	return max_time

'''Non-Operation'''
def MAX(t1, t2 , l):
	if (t1 >= t2):
		return t1, l[0]
	return t2, l[1]

''' Since the above two situtation happen separately we need to add the total time '''
''' The copy to DRAM from SRAM has to be sent 
	DRAM is single port and SRAM is dual port which implies that when the current_chunk is written to SRAM completely 
	We can either transfer the copy to DRAM first and then transfer the next set of chunk to CBUF from SRAM  or do the opposite either way there is no option 
	So this means that the time taken to transfer to DRAM must be added to the total time  '''

''' A level higher: L1 level''' 

'''Operation'''
def total_time_per_SRAM_chunk(direction,resnet_flag,cached_for_resnet,index_sram_chunk,index_cbuf_chunk):
	''' calculate the total time taken to complete a chunk present in SRAM '''
	
	global SRAM_clock_freq, Clock_freq
	check_sram_overflow(index_sram_chunk)
	pipe_2_3_time, output_CBUF_chunk = level_two_pipeline_cbuf_delivery(resnet_flag,index_sram_chunk,index_cbuf_chunk,cached_for_resnet)
	pipe_2_3_time = pipe_2_3_time/Clock_freq
	pipe_1_4_time = level_two_pipeline_sram_assembly(index_sram_chunk,index_cbuf_chunk)/Clock_freq
	copy_time = time_SRAM_DRAM()/SRAM_clock_freq
	
	time_first_chunk = (time_SRAM_CBUF(index_sram_chunk, index_cbuf_chunk) + time_CBUF_Assembly(index_sram_chunk, index_cbuf_chunk) + time_Assembly_Delivery(index_sram_chunk,index_cbuf_chunk))//Clock_freq   # time taken to compute  first chunk in CBUF and transfer to Delivery Group
	time_subsequent_chunks =  pipe_2_3_time + pipe_1_4_time + copy_time
	total_time_SRAM_chunk = 0  # time taken to empty the chunk that was stored in SRAM from DRAM, these chunks are mentioned in SRAM_CHUNKS_FROM_DRAM array
	'''first we compute the time to finish the chunk stored in SRAM '''
	logging.info("Computing time for SRAM chunk {}".format(index_sram_chunk))
	# for i in range(len(CBUF_CHUNKS_FROM_SRAM[index_sram_chunk])-1):        # eg -> index_sram_chunk =0 picks the first sram chunk that we will transfer to CBUF
	# 	''' at the end of this the first chunk in SRAM has been computed
	# 		to complete the layer we must do this process again for all the chunks that have to be transferred from DRAM to SRAM '''
	# 	total_time_SRAM_chunk += time_subsequent_chunks

	total_time_SRAM_chunk = time_subsequent_chunks + time_first_chunk
	logging.info("Total time to transfer : {}".format(total_time_SRAM_chunk))
	logging.info("At this point, chunk number {} stored in CBUF_CHUNKS_FROM_SRAM has been computed and the time to do so has been calculated.".format(index_cbuf_chunk))
	return total_time_SRAM_chunk, output_CBUF_chunk

'''Operation'''
def total_time_per_layer(direction,resnet_flag, index_input,cached_for_resnet):
	'''calculate the total time to finish one layer'''
	# add exception handling here later
	global previous_output_in_SRAM, weights , feature, GDDR6_clock_freq
	total_time_layer = 0
	total_chunks_from_DRAM_SRAM = length_SRAM_CHUNKS_FROM_DRAM(index_input)
	total_chunks_from_SRAM_CBUF_per_SRAM_chunk = length_CBUF_CHUNKS_FROM_SRAM()
	
	print("---IN total_time_per_layer()---")
	for i in range(total_chunks_from_DRAM_SRAM):
		logging.info("Computing time for SRAM chunk {}".format(i))
		total_time_layer += time_DRAM_SRAM(direction, index_input,i)/GDDR6_clock_freq
		for j in range(total_chunks_from_SRAM_CBUF_per_SRAM_chunk[i]): # all the sram chunks and the associated sub-chunks for each of the sram chunks must be computed per layer	
			logging.info("Computing for chunk number {0} in CBUF from SRAM. Top level chunk number in SRAM from DRAM is {1}".format(j,i))
			time , data = total_time_per_SRAM_chunk(direction,resnet_flag,cached_for_resnet,i,j)
			total_time_layer += time
			#previous_output_in_SRAM += data
	previous_output_in_SRAM = feature[index_input + 1] 
	print("This is the previous output in SRAM",previous_output_in_SRAM, " index_input: ",index_input)
	logging.info("Total time to transfer (---IN total_time_per_layer()---): {}".format(total_time_layer))
	logging.info("At this point layer: {} has been computed and the time to do so has been calculated.".format(index_input))
	return total_time_layer

'''Non-Operation'''
def total_layers_in_network():
	''' calculate total convolution layers in the network '''
	counter = 0
	with open('weight-size.txt','r') as f:
		for line in f:
			counter += 1
	return counter


total_inference_time = 0  # total image inference time

'''Operation'''
def total_inference_time():
	'''calculate the total inference time 
	   total number of convolution layers in the Network
	   Resnet configured every 2 layers , not the case in YOLOv3(fix this later)'''
	global SRAM_CHUNKS_FROM_DRAM,CBUF_CHUNKS_FROM_SRAM,Clock_freq, previous_output_in_SRAM

	layers = total_layers_in_network()
	logging.info("Number of layers in Network {}".format(layers))
	time_inference = 0
	resnet_flag = 0
	cached_for_resnet = 0
	logging.info("Calculating time for complete inference of image input")
	for layer in range(layers):
		if (layer%2 == 0 and layer > 4 and layer < 51):
			resent_flag = 1
			cached_for_resnet = feature[layer-2]
			logging.info("LAYER NUMBER : {}".format(layer + 1))
			logging.info("Layer size: {}".format(weights[layer]+ feature[layer]))
			
			logging.info("Resnet flag: {0} , Cached output: {1}".format(resent_flag, cached_for_resnet))
			layer_time = total_time_per_layer('right',resnet_flag,layer,cached_for_resnet)  
			time_inference += layer_time
			logging.info("Completed Layer..")
		else:
			resnet_flag = 0
			logging.info("LAYER NUMBER : {}".format(layer))
			logging.info("Resnet flag: {0}".format(resnet_flag))
			logging.info("Layer size: {}".format(weights[layer]+ feature[layer]))
			
			layer_time = total_time_per_layer('right',resnet_flag,layer,cached_for_resnet)   
			time_inference += layer_time
			logging.info("Completed Layer..")
		SRAM_CHUNKS_FROM_DRAM = []      # reinitialize after finishing a layer
		CBUF_CHUNKS_FROM_SRAM = []  	# ---------------"--------------------	  # ---------------"--------------------
	time_softmax = softmax()   			# assuming cpu clock freq = 2.7Ghz, running on 2.7 GHz Intel Core i5, 8 GB 1867 MHz DDR3 
	time_upsampling = upsampling()  	# ------------------------------------------"""------------------------------------------
	total_inference_time = time_inference + time_softmax + time_upsampling  # assuming cpu and nvdla run at same clock frequency
	logging.info("TOTAL INFERENCE TIME (in sec): {}".format(time_inference))
	#logging.info("Time : {}".format(time_sec))
	logging.info("UPSAMPLING TIME: {0}	SOFTMAX: {1}".format(time_upsampling, time_softmax))
	logging.info("TOTAL INFERENCE TIME (CLK FREQ = 2.7GHz) (in seconds): {}".format(total_inference_time))
	logging.info("COMPLETED INFERERENCE ON IMAGE SUCCESSFULLY!!!")

'''Non-Operation'''
def update_sram(index_sram_chunk):
	''' update the max size of data stored in sram'''
	global SRAM_size_in_bits , SRAM_CHUNKS_FROM_DRAM, data_size_Delivery_Group
	input_percentage = (SRAM_CHUNKS_FROM_DRAM[index_sram_chunk]/SRAM_size_in_bits)*100
	output_percentage = (previous_output_in_SRAM/SRAM_size_in_bits)*100
	remaining_percentage = 100 - input_percentage - output_percentage
	return input_percentage, output_percentage, remaining_percentage

'''Non-Operation'''
def check_sram_overflow(index_sram_chunk):
	''' this function tracks the data in SRAM during processing 
		Reports overflow and exits if such an event occurs 
		Otherwise returns filled space and percentage of filled space per data type (input, weight, output) 
		Further we could provide chunk specific and associated input, weight, output values '''
	global SRAM_size_in_bits
	inp , out, remaining = update_sram(index_sram_chunk)
	inp_size = (inp*SRAM_size_in_bits)/100
	out_size = (out*SRAM_size_in_bits)/100
	if(inp + out > 100):
		print(inp_size, " ", out_size)
		logging.info("Overflowing SRAM... Exiting for now...")
		sys.exit(0)
	logging.info("Space occupied in SRAM : {0}% by Input Data (feature + weight) and {1}% by Output from SDP. Empty space {2}%".format(inp,out,remaining))

''' check stack trace of function calls 
	
	main() -> total_inference_time() ->  '''

'''Non-Operation'''
def optimizer():
	'''
	1) this functions simulates different combinations of SRAM_size, CBUF_size, MAC_cells to find an optimum (minimum inference time) inference time
		First we generate different combinations of the above three parameters and run the inference for all these combinations 
		Each combination will output an inference time. 
	So given a combination ---> time_to_inference , compare against base case if  time_inference_new < time_inference_base add this combination to the list
	compute derivative of T_inference function relative to the 3 parameters. 
	define error function = new_time_inference - base_case
	new_SRAM_size = SRAM_size - delta(error).alpha
	same for other three parameters 
	Like a perceptron :
		SRAM	------>------		 --------
							|		 |		|
							---------|Per	|
	   CBUF		------>--------------|cep	|----->---T_inference
							---------|tron:	|
						   |		 |F()   |
	   CMAC		------>----          --------

	Brute force method would be to generate as many possible combinations of the 3 parameters and find inference time for each of them 
	But the above one seems to be a smarter alternative.
	
	2) Given a desired inference time of T_inference_desired = 0.03 sec (say)
	   We can use the Perceptron model to find the right parameters for SRAM, CBUF,MAC. 

	'''
	# Check all combinations
	global SRAM_size_in_bits, size_atomic_op, total_inference_time
	sram_options = [8,10,12,14,16]*(1024*1024)
	atomic_op_options = [8704, 16896, 25088, 33280, 41472]
	inference_time_array = {}
	#total possible combimations for the two lists = 5C1*5C1
	total_combinations = 25
	for i in range(len(sram_options)):
		for j in range(len(atomic_op_options)):
			SRAM_size_in_bits = sram_options[i]
			size_atomic_op = atomic_op_options[j]
			total_inference_time()
			inference_time_array.update({total_inference_time : list(SRAM_size_in_bits,size_atomic_op) })

	
	logging.info("Inference time Array: {}".format(inference_time_array))

	

def main():
	'''call the total_inference_time() function to initiate the inference process '''
	total_inference_time()
	#optimizer()

if __name__ == "__main__":
	start = timeit.timeit()
	main()
	end = timeit.timeit()
	logging.info("Program Run time (Not related to inference time): {}".format(end-start))