def rcnn_computation():
	''' this is just to see how RCNN layers would be computed '''
	''' 2 convolution operations + 1 Max pooling ( 2 times )
		3 convolution operatons  + 1 Max pooling (2 times )
	'''
	for i in range(1,18):
		if (i%3 == 0 and i<=6):
			print("MAx pooling... " + "Layer: {}".format(i))
		elif (i == 10 or i == 14):
			print("MAx pooling..." + "Layer: {}".format(i))
		else:
			print("convolution operation with RELU..." + "Layer: {}".format(i))

if __name__ == '__main__':
	rcnn_computation()
