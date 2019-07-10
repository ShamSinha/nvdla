#!/usr/local/bin/python3
''' matrix concatenation along different dimensions '''
import random
import sys
def create_matrix():
	''' define 5d matrices   '''
	skip = 1
	mat_1 = [[[[[round(random.random(),3) for i in range(1)] for j in range(1)] for k in range(64)] for l in range(18)] for m in range(2)]
	mat_2 = [[[[[0 for i in range(1)] for j in range(1)] for k in range(64)] for l in range(18)] for m in range(2)]
	mat_3 = [[[[[0 for i in range(1)] for j in range(1)] for k in range(64)] for l in range(1)] for m in range(2)]
	check_type(skip,mat_1)
	check_dims(mat_1, mat_2)
	check_dims(mat_1, mat_3)

	return mat_1, mat_2, mat_3

i = 1
def check_dims(m_1, m_2):
	''' check for right dimension size '''
	global i
	dims_1 = dimensions(m_1)
	dims_2 = dimensions(m_2)
	try:
		d_1 = len(dims_1)
		d_2 = len(dims_2)
		if d_1 == d_2:
			#concatenation is possible
			print("Concatenation " + str(i) + " can proceed")
			i += 1
		else:
			raise Exception("The dimensions dont match can't concatenate..")
	except Exception as e:
		print(e.message, e.args)
 

def matrix_concat_selection():
	''' choose which side to concatenate along '''
	choice_ques = ''' Which side to concatenate along....
	1) Press c for channel direction
	2) Press l for left side direction
	3) Press r for right side direction
	4) Press t for top side direction 
	5) Press b for bottom side direction 
	6) Press e to exit this exercise 
	Enter choice = '''
	choice = input(choice_ques)
	return choice

def matrix_concat(c,m_1,m_2,m_3):
	''' perform concatenation based on choice '''
	if (c == 'c'):
		print("Concatenating along channel direction")
		concat_mat = channel(m_1,m_2)
	elif(c == 'l'):
		print("Concatenate along left direction")
		concat_mat = left(m_1, m_3)
	elif(c == 'r'):
		print("Concatenate along right direction")
		concat_mat = right(m_1, m_3)
	elif(c == 't'):
		print("Concatenate along top direction")
		concat_mat = top(m_1, m_3)
	elif(c == 'b'):
		print("Concatenate along bottom direction")
		concat_mat = bottom(m_1, m_3)
	
	return concat_mat

def channel(m,n):
	d_m = dimensions(m)
	zero_concat = [[ [] for i in range(d_m[3])] for j in range(d_m[4])]
	
	for c_n in range(d_m[4]):
		for a_n in range(d_m[3]):	
			li = m[c_n][a_n][:][:][:]
			ap = n[c_n][a_n][:][:][:]
			li.extend(ap)
			zero_concat[c_n][a_n].extend(li)
	
	return zero_concat

def dimensions(mat):
	'''find the dimension lengths '''
	zero_th =len(mat[:][0][0][0][0])  # width axis
	first =  len(mat[:][:][0][0][0])  # height axis
	second = len(mat[:][:][:][0][0])  # channel axis
	third =  len(mat[:][:][:][:][0])  # atom cube number per cube
	fourth = len(mat[:][:][:][:][:])  # cube number
	dims = [zero_th,first,second,third,fourth]
	return dims

def check_type(skip,mat):
	if skip:
		pass
	else:
		print(type(mat[:][:][:][0][0]))
		print(len(mat[:][:][:][0][:]))
		print(mat[0][0][:][:][:])
		print(mat[0][0][0][:][:])
		

def main():
	m_1, m_2, m_3 = create_matrix()
	choice = matrix_concat_selection()
	if choice == 'e':
		sys.exit(0)

	m_out = matrix_concat(choice,m_1,m_2,m_3)
	dims_out = dimensions(m_out)
	print(dims_out)
if __name__ == '__main__':
	main()