#!/usr/local/bin/python3
''' Define Data types used in NVDLA 
	int8 , int16 , fp16 , fp32 : are all the data types used in NVDLA 
	
	the Datatype class should construct the required datatype so it can be used similar to any other datatype such as int64
	or char in case of character letters 
'''


# GET BACK TO THIS LATER, UNDERSTANDING WHAT SIZE MEANS IS IMPORTANT 
# COULD USE sys.getsizeof() for the purpose of getting the size of the container + data pointed to inside the container

import sys

class Nvdladatatype():
	''' define the datatype for input and weight data used in NVDLA '''
	def __init__(self, precision, value):
		''' enter the precision in bits '''
		self.precision = precision
		self.value = value

	def get_zero_initalized_bytes(self):
		''' return the bytes based on the precision '''
		pres = self.precision//8
		b = bytes(pres)
		return b

	def get_byte_val_for_array(self):
		''' return the bytes with associated integer value '''
		b = bytes(self.value)
		return b

	def bit_print(self):
		print("Precision : {}".format(self.precision))
		print("Value in bytes : {}".format(self.get_zero_initalized_bytes()))
		print("Value in bytes for given list : {}".format(self.get_byte_val_for_array()))

	def generate_byte_value_list(self, in_list):
		if (self.precision  == 8):
			byte_list = bytes(in_list).hex()
			
		elif (self.precision == 16):

			byte_list = bytearray(in_list).zfill(4)
			byte_list = byte_list.hex()
		
		return byte_list

obj = Nvdladatatype(16, [12,3])
obj.bit_print()
in_list = [3,4,5,6,8,100,45]
returned_byte_list = obj.generate_byte_value_list(in_list)
print (returned_byte_list)