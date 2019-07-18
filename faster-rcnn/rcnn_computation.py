def main_routine():
	''' this is just to see how RCNN layers would be computed '''
	''' 2 convolution operations + 1 Max pooling ( 2 times )
		3 convolution operatons  + 1 Max pooling (2 times )
	'''
	for i in range(1,23):
		if (i%3 == 0 and i<=6):
			print("MAx pooling... " + "Layer: {}".format(i))
		elif (i == 10 or i == 14):
			print("MAx pooling..." + "Layer: {}".format(i))
		elif (i == 18):
			print("Subrountine beginning...")
			print("Convolution operation... No RELU. Output of this convolution should feed the two subrountine defined below" + "Layer: {}".format(i))
			resp_1 = subroutine_1('conv')
			print(resp_1)
			resp_2 = subroutine_2('conv')
			print(resp_2)
			print("Performing Proposal layer...")
		elif (i == 19):
			print("Using output of Subrountine_1 and Subrountine_2 to compute ROI_POOLING...")
		elif (i == 20 or i == 21):
			print("Performing Fully connected operation..." + "Layer: {}".format(i))
		elif(i == 22):
			print("Subroutine_3 beginnging... ")
			resp_3 = subroutine_3('conv')
			print(resp_3)
			print("Computing Bbox_predictions...")
			print("--------Completed Faster RCNN-----------")
		else:
			print("convolution operation with RELU..." + "Layer: {}".format(i))

def subroutine_1(conv_in):
	''' take the convolution input, perform the subrountine '''
	for i in range(4):
		if(i%2 != 0):
			print("Reshaping...")
		elif(i == 0):
			print("Convolution operation... No RELU" + "Layer: {}".format(i))
		else:
			print("Performing softmax operation... ")
	return ("		Subrountine_1 completed!")

def subroutine_2(conv_in):
	''' take the convolution input, perform the subrountine '''
	print("Performing Convolution operation.. No RELU")
	return ("		Subrountine_2 completed!")

def subroutine_3(conv_in):
	''' take input of previous layer and compute the class probabilites and scores '''
	print("Computing class scores...")
	print("Computing class probabilites...")
	return ("		Subrountine_3 completed!")

if __name__ == '__main__':
	main_routine()
