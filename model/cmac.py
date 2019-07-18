# Simulation of CMAC behaviour 

import datacube 

def matrixmul(n_mul, n_cells, precision):
	''' Simulate the matrix multiplication process in CMAC based on the documentation provided by NVDLA

	    Input:
	    	n_mul: number of multipliers per MAC cell
	    	n_cells: number of MAC cells
			precision: the precision of each data point

		Output: 
			Number of cycles to compute one Atomic Operation
	'''

	# for i in range(n_cells):
	# 	for j in range(n_mul):
	pass
