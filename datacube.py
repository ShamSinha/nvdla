#!/usr/local/bin/python3

''' Define the data cube, create the atomic cubes
    This will feed the CMAC eventually
	
    Right now we directly use this for CMAC computation

'''
import pprint
import random
class Datacube():
	atomic_5d = [[[[[]]]]]  # empty atomic cube data structure unique to every data cube 
	def __init__(self, width, height, channels, n_cubes=1):
		self.width = width
		self.height = height
		self.channels = channels
		self.n_cubes = n_cubes

	def define_precision(self):
		pass

	def dimensions(self):
		'''define the dimensions of atomic cubes across all cubes '''

		channel_size_atomic = 64 
		blocks = self.channels/channel_size_atomic
		n_atomic_per_cubes = int(self.width*self.height*blocks)
		n_atomic_all_cubes = int(n_atomic_per_cubes*self.n_cubes)
		
		return n_atomic_per_cubes, n_atomic_all_cubes

	def zero_concat(self):
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
			zero_concat_channels = self.zero_concat()
			print(zero_concat_channels)

			self.atomic_5d = [[[[[random.random() for i in range(atomic_width)] for j in range(atomic_height)] for k in range(channel_size_atomic)] for l in range(zero_concat_channels)] for m in range(self.n_cubes)]

		return self.atomic_5d

	def print_values(self):
		'''print the dimensions of the data cube '''

		print(self.width, " ", self.height, " ", self.channels, " ", self.n_cubes)


def ask_user():

	w = int(input("Enter width of data cube"))
	h = int(input("Enter height of data cube"))
	c = int(input("Enter channel size of data cube"))
	choice = input("Enter choice of data cube (only for kernel press 'k')")
	if (choice == 'k'):
		n = int(input("Enter number of cubes (relevant for kernel only, defaults to 1)"))
	else:
		n =1

	input_cube = Datacube(w,h,c,n)
	input_cube.print_values()
	atmoics = input_cube.initialize_atomic_cubes()
	pprint.pprint(atmoics[:][:][:][0][0])
	per , total = input_cube.dimensions()
	print("Number of atomic cells per cube: {}".format(per))
	print("Number of atomic cells in total: {}".format(total))
	print('channel dimension length of atomic cubes for this input cube : {}'.format(len(input_cube.atomic_5d[:][:][:][0][0])))


if __name__ == '__main__':
	ask_user()