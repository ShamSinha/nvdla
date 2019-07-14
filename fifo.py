''' FIFO structure implementation for the various buffers used in NVDLA 
	A continous storage unit that is accessed in a First-In-First-Out basis

	Use as follows: Define an instance of FIFO class giving it the name of the module inside NVDLA that uses it
	Ex. bdma_fifo = Fifo(`parameters`) for the BDMA module '''
import sys
import logging

# config for logging. Log in GDDR6.log, check for debugging
logging.basicConfig(filename="fifo.log", filemode='w',format='%(name)s - %(levelname)s - %(message)s')


class Fifo():
	''' Enter parameters for the buffer based on requirement 
		length = depth of the FIFO
		size_of = size of each buffer unit 

		fifo = buffer FIFO that we are interested in 
	'''
	def __init__(self, length, size_of):
		self.length = length
		self.size_of = size_of
		self.fifo = self.get_fifo(self.length, self.size_of)

	def get_fifo(self, l, s):
		fifo = [0 for i in range(0,l)]
		return fifo

	def print_queue(self):
		'''print the buffer as it is now '''
		print(self.fifo)

	def enqueue(self,obj):
		''' obj entered should be of the right size as allowed by self.size_of 
			Configure this later '''

		if (is_full()):
			overflow()
		else:
			self.fifo.insert(0,self.obj)
			del self.fifo[len(self.fifo) - 1]

	def dequeue(self):
		if (is_empty()):
			underflow()
		else:
			dequeued_element = self.fifo.pop(len(self.fifo) -1 )

		return dequeued_element

	def overflow(self):
		logging.error("Buffer overflow...")
		sys.exit(1)

	def underflow(self):
		logging.error("Buffer underflow..")
		sys.exit(1)

	def get_length_of_buffer():
		return self.length

	def filled_length(self):
		''' non-zero elements in the buffer rightnow '''
		counter = 0
		for val in self.fifo:
			if val != 0 :
				counter += 1
		return counter

	def is_empty(self):
		filled_value = self.filled_length()
		if filled_value == 0:
			return True
		else:
			return False
	
	def is_full(self):
		filled_flag = self.filled_length()
		if filled_length == self.length :
			return True
		else:
			return False



