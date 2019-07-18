#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
# DO THIS FOR WEIGHT DATA AS WELL
import sys
import logging
import random
# # gddr and sram have different clocks frequency
# Memory_size = 8589934592   #8Gb

''' 
	Cycles + number of reads from GDDR6 for input data and weight data transfer to SRAM (on-chip)
	
	Input data = H*W*C
	Weight data = R*S*C*K

	mapping in GDDR6 of all the data is described as follows :
		GDDR6 has 2 channels 
		Each channels is a r*c matrix of cells where 
		r = rows 
		c = coloumns 
		cell_size(cell size) = 256 bits or 32 Bytes 
		channels = 2 
		and each channels has width = 256 bit 
		a parallel to serial converter translates this 256 bits into 16 bit word and there are 16 such words transferred in sequence 
		same is done for the other channel 

		For each of GDDR6 read operation we transfer 256 bits per channel or 512 bits in total 

		Now the input data is 416*416*3*8 bits of data 
		So,
			First lets map input data into GDDR6 then we can access it 512 bits at a time which will allow us to calculate the total cycles taken to 
			complete the access of input data 
			Do the same for weight data 

'''
# config for logging. Log in GDDR6.log, check for debugging
logging.basicConfig(filename="GDDR6.log", filemode='w',format='%(name)s - %(levelname)s - %(message)s')

def input_data_GDDR6_channel_mapping(H,W,C,r,c,cell_size):
	'''this will map input data to GDDR6 '''
	precision = 8
	total_size_input_data = H*W*C*precision
	cell_req = total_size_input_data/cell_size
	cell_avaiable = r*c  # per channel of GDDR6
	no_of_banks = 16
	cells_per_bank = c*(r/no_of_banks)
	size_of_bank = cells_per_bank*cell_size
	banks_required = cell_req/cells_per_bank

	if (cell_req > cells_per_bank):
		logging.warning("data overflows bank")
		if (banks_required > 16)
			logging.error("Overflow... Exiting")
			# log error here 
			# exit with error code
			sys.exit(1)
		else:
			logging.info("More than one banks were used to store data")
	else:
		print("data was mapped into channel successfully")
		logging.info("data was mapped into channel successfully into channel")

	return cell_req


def split_between_channels(cells_required):
	''' since we have two channels and each has its own PtoS converter 
		we can transfer 16*16 + 16*16 words parallely to BDMA '''
		split_between_two_channel_factor = 2
		return split_between_two_channel_factor


def number_of_sequences(cells):
	''' generate the sequence for input data 
		return
		 the list of list of sequences generated
		 the number of such sequences required for the entire input data '''

	# 256 bits will be read from the channel and passed to the Parallel to Serial converter
	# 1 cell is read every time and passed into Parallel to Serial converter
	number_of_words = 16  # one sequence length
	size_of_each_word = 16 #bits
	elements_per_word = 2 # (8 + 8 ) bits

	total_number_of_sequences = cells

	list_of_sequences_generated = [[random.randint() for k in range(0,16)] for i in range(cells)]

	return (list_of_sequences_generated, total_number_of_sequences)


def time_to_read_data_from_GDDR6(column,cells_required,total_sequences):
	''' return the time taken to access one sequence (read time of GDDR6)
		REMEMBER TO USE CLOCK FREQUENCY OF GDDR6 HERE WHEN MULTIPLYING THE CYCLES OF reading_time FOR
		TIME IN SECONDS '''

	tck =2/3
	tRCD=10   # in tck = 2/3ns  # row access time
	tCCD = 3 					# time delay between two read operations 
	tRTP = 4 					# precharge time 	
	tRP = 22.5					# difference between activate and precharge ( min)
	tCL =18						# read latency ( count for every row )

	rows_utilized = cells_required//64 + cells_required%64

	refresh_time = (tRCD+(column-1)*tCCD+tRTP+tRP)/rows_utilized
	reading_time = rows_utilized*(tRCD+(column-1)*tCCD + tRTP + tRP+ tCL) + Refresh_time # cycles
	reading_time = reading_time/split_between_channels(cells_required)

	return reading_time












