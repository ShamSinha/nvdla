#!/usr/local/bin/python3

''' Define the data cube, create the atomic cubes
    This will feed the CMAC eventually
	
    Right now we directly use this for CMAC computation

'''
import pprint
import random
import matrix_concat 
import math
class Datacube():

	atomic_5d = [[[[[]]]]]  
	
	def __init__(self, width, height, channels, pres, pad=[0,0,0,0], n_cubes=1 ):
		self.width = width
		self.height = height
		self.channels = channels
		self.n_cubes = n_cubes
		self.pres = pres
		self.pad = pad
	
	def define_precision(self):
		if self.pres == 'int8':
			print('{:08b}'.format(10)) # use 8 bits to represent a number
		elif self.pres == 'int16':
			print('{:016b}'.format(10)) # use 16 bits to represent a number
		elif self.pres == 'fp16':
			pass 
		elif self.pres == 'fp32':
			pass
		else:
			print("Invalid choice. Not supported!!")
	
	def dimensions(self):
		'''define the dimensions of atomic cubes across all cubes '''

		channel_size_atomic = 64 
		blocks = math.ceil(self.channels/channel_size_atomic ) # FIX here, n_atomic_* go to 0 when channel size < 64
		n_atomic_per_cubes = int(self.width*self.height*blocks)
		n_atomic_all_cubes = int(n_atomic_per_cubes*self.n_cubes)
		
		return n_atomic_per_cubes, n_atomic_all_cubes

	
	def zero_concat_channels(self):  
		''' If channel size is not a multiple of 64 we append zeros 
			this function will generate the appropriate number of zero channels to be added
		'''
		for i in range(64):
			zero_concat_channels = self.channels + i
			if(zero_concat_channels%64 == 0):
				return zero_concat_channels
			else:
				continue
			return self.channels  
	
	
	def zero_concat_cube(self):
		'''return zero data cube '''
		atomic_width = 1
		atomic_height = 1
		channel_size_atomic = 64 
		zero_concat_channels = self.zero_concat_channels()  # number of channels with zero values
		#print(zero_concat_channels)
		zero_cube = [[[[[0 for i in range(atomic_width)] for j in range(atomic_height)] for k in range(channel_size_atomic)] for l in range(zero_concat_channels)] for m in range(self.n_cubes)]

		return zero_cube

	def sum_cubes(self, zero, atomic):  # Bug in function 
		''' concatenate two atomic data cubes when the channel dimension is not a multiple of 
			64 
			input:
				original atomic cube = atomic
				zero data atomic cube = zero
			output:
				zero concatenated atomic cube = atomic 
		'''
		# Addition will take place in the channel dimension all things being the same
		blocks, x = self.dimensions()
		surface_blocks = self.height*self.width
		blocks = blocks//surface_blocks
		cube_blocks = surface_blocks*blocks
		#zero_concat = [[ [] for i in range(surface_blocks)] for j in range(self.n_cubes)]
		for c_n in range(self.n_cubes):
			for a_n in range(surface_blocks):
				li = atomic[c_n][cube_blocks - a_n-1][:][:][:]
				ap = zero[c_n][a_n][:][:][:]
				print('That')
				li.extend(ap)
				atomic[c_n][cube_blocks - 1 - a_n].extend(li)
				
		return atomic 

	def initialize_atomic_cubes(self):
		'''initialize the atomic data cubes given the data cube dimensions
			input :
				data cube dimensions
			output:
				number of atomic cubes
				Zero initailzed values of each atomic cube
				append zeros in case c%64 != 0
		'''
		atomic_width = 1
		atomic_height = 1
		channel_size_atomic = 64 
		blocks = self.channels/channel_size_atomic
		n_atomic_per_cubes = int(self.width*self.height*blocks)
		n_atomic_all_cubes = int(n_atomic_per_cubes*self.n_cubes)
		
		if blocks.is_integer():
			self.atomic_5d = [[[[[random.random() for i in range(atomic_width)] for j in range(atomic_height)] for k in range(channel_size_atomic)] for l in range(n_atomic_per_cubes)] for m in range(self.n_cubes)]
			
		else:
			zero_cube = self.zero_concat_cube()
			self.atomic_5d = [[[[[random.random() for i in range(atomic_width)] for j in range(atomic_height)] for k in range(channel_size_atomic)] for l in range(n_atomic_per_cubes)] for m in range(self.n_cubes)]
			self.atomic_5d = self.sum_cubes(zero_cube, self.atomic_5d)

			# zero_concat_channels = self.zero_concat_channels() 
			# self.atomic_5d = [[[[[random.random() for i in range(atomic_width)] for j in range(atomic_height)] for k in range(channel_size_atomic)] for l in range(zero_concat_channels)] for m in range(self.n_cubes)]
		return self.atomic_5d

	
	def print_values(self):
		'''print the dimensions of the data cube '''
		print(self.width, " ", self.height, " ", self.channels, " ", self.n_cubes)
		print(self.pad[0], " ", self.pad[1], " ", self.pad[2], " ", self.pad[3])

def ask_user():
	''' test code for above'''

	try:
		w = int(input("Enter width of data cube"))
		h = int(input("Enter height of data cube"))
		c = int(input("Enter channel size of data cube"))
		choice = input("Enter choice of data cube (only for kernel press 'k')")
		if (choice == 'k'):
			n = int(input("Enter number of cubes (relevant for kernel only, defaults to 1)"))
		else:
			n =1
		pres = input("Enter the precision")
		choice = input("Padding?? (y/n)")
		if(choice == 'y'):
			left = int(input("Enter left padding "))
			right = int(input("Enter right padding"))
			top  = int(input("Enter top padding"))
			bottom = int(input("Enter bottom padding"))
		else:
			[left,right,top,bottom] = [0,0,0,0]
		pad = [left, right, top , bottom]
		input_cube = Datacube(w,h,c,pres,pad,n)
		input_cube.print_values()
		atmoics = input_cube.initialize_atomic_cubes()
		pprint.pprint(atmoics[:][:][:][0][0])
		per , total = input_cube.dimensions()
		print("Number of atomic cells per cube: {}".format(per))
		print("Number of atomic cells in total: {}".format(total))
		print('channel dimension length of atomic cubes for this input cube : {}'.format(len(input_cube.atomic_5d[:][:][:][0][0])))

		
	except Exception as e:
		print ("Some error occured")


if __name__ == '__main__':
	ask_user()