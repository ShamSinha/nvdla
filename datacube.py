''' Define the data cube
    This will feed the CMAC eventually

    Right now we directly use this for CMAC computation

'''
import pprint

class Datacube():

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
		print("Number of atomic cells per cube: {}".format(n_atomic_per_cubes))
		print("Number of atomic cells in total: {}".format(n_atomic_all_cubes))

	def zero_concat(self):
		
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
		'''
		atomic_width = 1
		atomic_height = 1
		channel_size_atomic = 64 
		blocks = self.channels/channel_size_atomic
		n_atomic_per_cubes = int(self.width*self.height*blocks)
		n_atomic_all_cubes = int(n_atomic_per_cubes*self.n_cubes)
		
		if blocks.is_integer():
			atomic_3d = [[[[[0 for i in range(atomic_width)] for j in range(atomic_height)] for k in range(channel_size_atomic)] for l in range(n_atomic_per_cubes)] for m in range(self.n_cubes)]
		else:
			zero_concat_channels = self.zero_concat()
			print(zero_concat_channels)
			atomic_3d = [[[[[0 for i in range(atomic_width)] for j in range(atomic_height)] for k in range(channel_size_atomic)] for l in range(zero_concat_channels)] for m in range(self.n_cubes)]

		return atomic_3d

	def print_values(self):
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



if __name__ == '__main__':
	ask_user()