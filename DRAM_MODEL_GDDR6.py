#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 02:23:51 2019

@author: Shubham
"""


# gddr and sram have different clocks frequency
Memory_size = 8589934592   #8Gb

#READING TIME FROM DRAM
tck =2/3
tRCD=10   # in tck = 2/3ns  # row access time
tCCD = 3 					# time delay between two read operations 
tRTP = 4 					# precharge time 	
tRP = 22.5					# difference between activate and precharge ( min)
tCL =18						# read latency ( count for every row )

Num_channel = 2
prefetch = 16
word_size = 16  #in bits
BPE = 1   #for int8

H = 416
W = 416
C =3


data_size= H*W*C*8*BPE  # in bits

#per channel

Bank_group =4
num_banks = 16
Array_prefetch = prefetch*word_size
num_col = 64
num_rows = 16384

row = data_size/(Array_prefetch*Num_channel*num_col)
print(row)

Refresh_time = (tRCD+(num_col-1)*tCCD+tRTP+tRP)/row
reading_time = row*(tRCD+(num_col-1)*tCCD + tRTP + tRP+ tCL) + Refresh_time

bank_rows = num_rows // 16
bank_cols = num_col
cell_size = 32
in_gb = 1024*1024
bank_size  = (cell_size*bank_rows*bank_cols)// in_gb

print(bank_size)
print(reading_time)














