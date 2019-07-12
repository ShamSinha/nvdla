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

The wait time in cases where multiple read/writes must happend with a memory unit, can be modelled as a stochastic varible. The specifics can be hammered later

The goal is that the program outputs an equation for computation of an image input. This can be done only when all the layers in the network have been computed. 


									
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
		total_chunks_to_read = chunks_to_be_read/SRAM_writing_bandwidth   # total number of chunks we need to read from SRAM
		total_transfer_time = 0
		first_transfer_time = SRAM_writing_time + delay_cdma + CBUF_Reading_time
		logging.info("SRAM --> CBUF")
		
		for chunk in range(total_chunks_to_read-1): # for the remainder of chunks we just add the time taken to read from SRAM
			total_transfer_time += SRAM_reading_time

		total_transfer_time = total_transfer_time + first_transfer_time
		logging.info("Total time to transfer : {}".format(total_transfer_time))
		logging.info("At this point first chunk of data stored in CBUF_CHUNKS_FROM_SRAM has been transferred to CBUF. This fills CBUF")

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

size_atomic_op = 8704  #bits is the total size of input-- 1x1x64 + kernel--1x1x64x16 while considering int8 precision 
delay_csc = t0_CSC
delay_cmac = t0_CMAC
delay_adder_array = t0_CACC_Adder

Assembly_reading_time = t0_Assembly
Assembly_writing_time = t1_Assembly
Delivery_reading_time = t0_Delivery
Delivery_writing_time = t1_Delivery

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
	logging.info("At this point first chunk of data stored in CBUF_CHUNKS_FROM_SRAM has been computed and stored in Assembly_Group. This empties CBUF")
	
	return total_transfer_time


delay_truncation = t0_truncation
size_partial_sum_generate_per_atomic_op = 544 # bits in case of int8
Assembly_writing_bandwidth = Assembly_writing_bits 
Assemby_data_size = 0
''' Now that the result of the first chunk that was transferred to CBUF has been computed and result stored in CACC_Assembly_Group, we can begin to unload 
	this data into Delivery Group. Total time taken to complete this procedure will be added to the inference time '''

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
	global Assembly_writing_bandwidth, delay_truncation, size_partial_sum_generate_per_atomic_op,CBUF_CHUNKS_FROM_SRAM,size_atomic_op
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
	logging.info("At this point first chunk of data stored in CBUF_CHUNKS_FROM_SRAM has been computed and stored in Delivery Group. This empties Assembly")
	
	return total_transfer_time

''' Now that the Delivery Group is filled up we can start sending the to SDP for post- processing. Subsequenlty this data would be put back in on-chip SRAM

Important Note: Since the Assembly is free, the Pipe 2 can also function concurrently with Pipe 3, here we have the option to pick which one is taking longer and start forming 
the L1_Pipe. '''

delay_sdp = t0_SDP
delay_sdp_resnet = t1_SDP
Delivery_Group_bandwidth = 512 # bits

def time_Delivery_SRAM(resnet_flag):
	''' calculate the time taken to transfer all data in Delivery SRAM to on-chip SRAM
		Before that we calculate the size of data in Delivery Group. 
		Since data stored in Assembly Group is int34 for int8 and int32 in Delivery 
		We can calculate the size of data in Delivery Group through simple unitary method

		takes resent_flag as input to determine if resnet operation needs to be computed. 
	'''
	global Assemby_data_size,Delivery_Group_bandwidth,Delivery_writing_time
 	
 	data_size_Delivery_Group = (Assemby_data_size*32)/34
 	total_reads_from_delivery = math.ceil(data_size_Delivery_Group/Delivery_Group_bandwidth)
 	logging.info("Size of Delivery_Group {}".format(data_size_Delivery_Group))
	logging.info("Delivery_Group ---> SDP ---> SRAM")
	if(!select_resnet(resnet_flag))
		logging.info("Non Resnet Layer.. Proceeding..")
	 	time_first_read_from_delivery = Delivery_writing_time + delay_sdp + SRAM_reading_time
	 	total_transfer_time = 0

	 	for i in range(total_reads_from_delivery-1):
	 		total_transfer_time += Delivery_writing_time

	 	total_transfer_time = total_transfer_time + time_first_read_from_delivery

		logging.info("Total time to transfer : {}".format(total_transfer_time))
		logging.info("At this point first chunk of data stored in CBUF_CHUNKS_FROM_SRAM has been computed and we are now transferring this from Delivery Group back to SRAM. This empties Delivery Group")
		return total_transfer_time
	else:
		logging.info("Resnet Layer.. Proceeding..")
		'''if resnet operation has to be performed we will wait for that to have to switch to delay_sdp_resnet 
		   Also at this point the DRAM would be transferring layer-2 values from MCIF to SDP buffer. This means that a separate pipeline would 
		   be established which would require to transfer from DRAM -->SRAM and we will have to compute the time it takes for this operation to happen 
		   Is it going to be parallel to Pipe1 + Delivery --> SDP 
		   Also be careful that we would also be writing back to DRAM the copy of current layer being computed so there could a wait or lag associated with this. 
		   MCIF can only write to or read from GDDR6 at a time. This means that there could be a wait period associated with resnet layers. Just think about this a little more
		'''

def select_resnet(flag):
	if (flag == 1):
		return True
	return False































def main():
	maxi = size_SRAM()
	timee = time_DRAM_SRAM('right',0,0)

if __name__ == "__main__":
	main()