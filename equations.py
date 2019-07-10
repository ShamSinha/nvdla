#!/usr/local/bin/python3

def amdahls_law(execution_old,fraction, speedup):
	'''Given the fraction of time the speedup works in the execution of the input data 
	   we can calculate the performance improvement of NVDLA computing module
	   Input : fraction of time 
	   		   speed up 
	   Output: performance improvenemt over previous setup

	'''
	execution_new = execution_old *((1-fraction) + (fraction/speedup))
	print("New execution time is : {0:.2f} milliseconds".format(execution_new))
	print("Overall speedup based on enhanced portion : {0:.2f}".format(execution_old/execution_new))
	print("====================================================")
	print("         ")


def desired_time(old,fraction):
	'''Compute the speedup required for the enhanced portion of the pipeline '''
	desired = 30  # 30 milliseconds to compute input data
	ratio = desired/old
	speedup_req = fraction/(ratio - 1 + fraction)
	print("The speedup necessary for desired execution time : {0:.2f} milliseconds".format(speedup_req))

def cpu_time(ip,cpi,rate):
	''' Compute the CPU time, modified for NVDLA compute module
		ip = 16 kerenel atomic cubes + 1 input atomic cube
		cpi = 1 atomic operation per cycle = 1
		rate = 1 GHx clock rate
	'''
	time = ip*cpi*rate
	print("The cpu time or NVDLA compute time is : {0:.2f}".format(time))

if __name__ == '__main__':
	old = 68
	fraction = 1/7
	speedup = 4
	amdahls_law(old, fraction,speedup)
	#desired_time(old,fraction)