#!/usr/local/bin/python3
import sys 
import math
import logging 
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

In the code following this introduction Functions are labelled as :
	Operation: if time is being calculated there
	Non-Operation : if time is not being calculated there these would include condition checker, other calculators etc.

In each operation function the state of the system is defined in terms of active or idle 
This is pretty important as it will tell us what modules are functional at any time 

Important Note - time in this file means cycles. 										
 '''

# lists holding the sizes in bits
weights = []
feature = []
total = []

logging.basicConfig(filename="pipe.log", 
                    format='%(asctime)s %(message)s',filemode='w') 
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

t0_GDDR6_512bits,t1_GDDR6_512bits,gddr6_clk,t0_SRAM_512bits,t1_SRAM_512bits,sram_clk,t0_BDMA_20Deep=(10,20,30,40,50,60,70)  # dummy values used for now for debugging

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


''' Lets us first try to achieve some answer for just one layer of a neural network 
	This would be followed by second layer and the third one which will encompass the Resnet Layer
	Once the three layers are done the same process will repeat continously until the entire network is completed
	Remember to commit after each layer completion 
'''

GDDR6_writing_time = t0_GDDR6_512bits  # time taken to write 512 bits of data to GDDR6
GDDR6_reading_time = t1_GDDR6_512bits  # time taken to read 512 bits of data from GDDR6
GDDR6_clock_freq = gddr6_clk

SRAM_size = size_SRAM() + 2 #MB
SRAM_writing_time = t0_SRAM_512bits    # time taken to write 512 bits of data to SRAM
SRAM_reading_time = t1_SRAM_512bits    # time taken to write 512 bits of data to SRAM
SRAM_clock_freq = sram_clk

delay_bdma = t0_BDMA_20Deep			   # time spent in bdma (useful for first transfer) difference between how long it takes for the first entry of sequence from GDDR6 to cross BDMA

SRAM_CHUNKS_FROM_DRAM = []  # stores the values for each chunk that will be transferred from DRAM --> SRAM

'''Operation'''
def time_DRAM_SRAM(direction, index_feature , index_weights):
	''' check for the direction in which the data should be transferred 
		right means DRAM -> SRAM
		left  means SRAM -> DRAM
	
	Note : DRAM  ------ active 
		   SRAM  ------ active
		   Other ------ idle   
	'''
	dummy_time = 0 # garbage time (throw away later)
	
	stages = {'dram':'active', 'sram':'active', 'cbuf':'idle', 'cmac':'idle', 'Assembly':'idle', 'Delivery':'idle', 'SDP':'idle'}
	logger.info("Active and Idle stages : {}".format(stages))
	
	if direction == 'right':
		logging.info("DRAM -> SRAM")
		global GDDR6_reading_time, weights, feature, SRAM_writing_time, delay_bdma
		total_transfer_time = 0
		total_data_to_transfer = feature[index_feature] + weights[index_weights] # total data to be transferred from DRAM -> SRAM 
		chunks_read_size = 512
		if can_sram_fit(total_data_to_transfer):
			''' At the end of this all data for convolution is going to present in SRAM now the DRAM is idle'''

			logger.info("SRAM can fit all data... Proceeding...")
			total_chunks_to_read = total_data_to_transfer//chunks_read_size + total_data_to_transfer%chunks_read_size
			transfer_time_first_chunk = GDDR6_reading_time + delay_bdma + SRAM_writing_time  # time taken to transfer just the first chunk of 512 bits 

			# for the remaining chunks
			for chunk in range(total_chunks_to_read-1):
				# for the remaining chunks, since Pipe 1 is established now, the time taken now would be only as fast as the slowest component i.e. GDDR6 read time
				# We could have multiplied, but for loop gives a better feel for transfer I think!
				total_transfer_time += GDDR6_reading_time

			# Now the total time taken can simply be added for first set of data to transfer into SRAM 
			total_transfer_time = total_transfer_time + transfer_time_first_chunk
			logging.info("Total time to transfer : {}".format(total_transfer_time))
			logging.info("At this point the data for Layer " + str(index_feature)+ " of YOLOv3 has been transferred to SRAM")
			return total_transfer_time
		
		else:
			''' since the whole data cannot be transferred to SRAM we need to split the data and transfer smaller amounts at a time 
				refer to the pseudo code to figure this out as there are certain other things that need to be taken care of before we decide how to and how much
				to split 
				Note that the core logic for calculating time would remain the same '''
			logger.info("SRAM cannot fit all data at once... Proceeding with smaller chunks...")
			sys.exit(0)

	elif direction == 'left':
		''' write to code for SRAM to DRAM transfer and get the total time in this case '''
		logger.info("SRAM -> DRAM")
		sys.exit(0)

	return dummy_time


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
		logger.info("SRAM_CHUNKS_FROM_DRAM holds just the total_size of feature + weights : {}".format(total_size))
		SRAM_CHUNKS_FROM_DRAM.append(total_size*denominator)
		return True
	else:  # make the sram_chunks fill up SRAM_CHUNKS_FROM_DRAM
		pass
	return False

''' Now we define the logic for transfer of data from SRAM to CBUF, 
	
	SRAM --> CDMA --> CBUF

	Since CBUF is smaller than SRAM we must be careful how much data is transferred at a time.
	Once the requisite amount of data has been transferred we can move forward with computation 
	This process of transfer of data from SRAM --> CBUF is going to happen multiple times 
	So first we calculate for the case when the first chunk is to be transferred '''

CBUF_CHUNKS_FROM_SRAM = [[]]  # stores the values for each chunk that will be transferred from SRAM --> CBUF 
							  # the index of CBUF_CHUNKS_FROM_SRAM would give us the SRAM_CHUNK that we are currently looking at 
CBUF_SIZE = 524288 # size of CBUF in bits 

''' Important Note - Everytime we process a layer the values arrays CBUF_CHUNKS_FROM_SRAM and SRAM_CHUNKS_FROM_DRAM need to be updated again. i.e. the arrays must be 
	emptied and refilled for the new layer. However while working on one layer there can be used as they are defined'''

'''Non-Operation'''
def make_chunks_SRAM_CBUF(index_weights,index_feature):
	''' create the appropriate chunks that will be transferred to CBUF for computation
		store the sizes in CBUF_CHUNKS_FROM_SRAM, this will be used later 
		SRAM_SUB_I --> CBUF_CHUNKS_FROM_SRAM where I are all the chunks we must transfer from SRAM to CBUF over time 
		Compute SRAM_SUB_I sizes. 
	'''
	global CBUF_CHUNKS_FROM_SRAM, weights, feature, SRAM_CHUNKS_FROM_DRAM, CBUF_SIZE
	
	for count,sram_chunk in enumerate(SRAM_CHUNKS_FROM_DRAM):
		''' for each sram_chunk in SRAM_CHUNKS_FROM_DRAM we need it split it further so that CBUF is filled all the time 
			except maybe in the last case when only remainder in SRAM would be transferred '''

		cbuf_chunk_size = CBUF_SIZE      # size of each cbuf_chunk that will be stored in cbuf_chunk_array that makes up one entry in CBUF_CHUNKS_FROM_SRAM
		remainder = sram_chunk%CBUF_SIZE # this is the last entry in cbuf_chunk_array 
		total_cbuf_chunks = math.ceil(sram_chunk/CBUF_SIZE )
		cbuf_chunk_array = []  # this list will store all the cbuf_chunk_sizes and this will be appended to CBUF_CHUNKS_FROM_SRAM

		for i in range(total_cbuf_chunks):
			if (i == total_cbuf_chunks-1):
				cbuf_chunk_array.append(remainder)
			cbuf_chunk_array.append(cbuf_chunk_size)

		CBUF_CHUNKS_FROM_SRAM.append(cbuf_chunk_array)   # CBUF_CHUNKS_FROM_SRAM will be created when all the indices in SRAM_CHUNKS_FROM_DRAM have been parsed
														 # Now, each individual values of CBUF_CHUNKS_FROM_SRAM will feed into the Pipe 2 


'''Operation'''
def time_SRAM_CBUF():
	''' calculate the total time taken to transfer one sram_chunk into CBUF 
		The sram_chunks stored in SRAM_CHUNKS_FROM_DRAM would be transferred into CBUF one at a time. Not adjacently. First one will be transferred to CBUF
		The time taken to do so would be calculated and then the next one would be calculated when the first one has been computed completely in the remaining pipeline '''
	


	






































def main():
	maxi = size_SRAM()
	timee = time_DRAM_SRAM('right',0,0)

if __name__ == "__main__":
	main()