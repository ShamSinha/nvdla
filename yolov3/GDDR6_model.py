tCCD = 2
tRTP = 2
tWL = 6
tCL = 15
tDQ = 2
tRCD = 10
tRP = 22.5     # 15 ns
tRC = 76.5     # 51 ns
tWR = 22.5     # 15 ns
t_refresh = 0
t_rest = 0
t_read = tRCD + 63*tCCD + tRTP + tRP +tCL
t_write = tRCD + 63*tCCD + tWL + tDQ + tWR + tRP

def create_rows(data_to_transfer):
	''' Input the data to be transferred '''
	pass

def read_time(data_size):
	'''calculate read time from GDDR6 for data to be transferred
	   Data to be transferred = n,  from GDDR6 to SRAM
	'''
	global t_rest, t_refresh, t_read
	t_sum = 0
	t_total = 0
	n = create_rows(data_size)
	for i in range(n+1):	#read one layer
		t_total = t_total + t_read	#no refresh
		if t_total > 2850:	#refresh judgement  
			t_left = 2850 - (t_total - t_read)	#how many tCK in left refresh row
			cell = 64-(t_left-10)/2-1	#how many cells in right refresh row
			if t_left < 10:	#refresh occurs in ACT
				t_refresh = t_refresh + 256.5
			elif cell == 0:	#refresh occurs in PRE
				t_refresh = t_refresh + 180
			else:	#refresh occurs in READ
				t_refresh = t_refresh + 204.5
			t_sum = t_sum + t_total - t_read + t_left + t_refresh
			t_total = 20 + cell*2	#the right time in refresh row
			t_refresh = 0
		if i == n:	#the last row add to sum
			t_sum = t_sum + t_total - t_rest
			t_rest = t_total
	return (t_sum)

def write_time(data_size):
	'''calculate write time for GDDR6 for data to be transferred 
	   Data to be transferred = n, from SRAM to GDDR6
    '''
    global t_rest, t_refresh, t_write
    t_sum = 0
    t_total = 0
    m = create_rows(data_size)
    for j in range(m+1):	#write one layer
		t_total = t_total + t_write#no refresh
		if t_total > 2850:	#refresh judgement
			t_left = 2850 - (t_total - t_write)	#how many tCK in left refresh row
			cell = 64-(t_left-10)/2-1	#how many cells in right refresh row
			if t_left < 10:
				t_refresh = t_refresh + 256.5
			elif cell == 0:
				t_refresh = t_refresh + 180
			else:
				t_refresh = t_refresh + 204.5
	
			t_sum = t_sum + t_total - t_write + t_left + t_refresh
			t_total = tRCD + tCCD*(cell-1) + tWL + tDQ + tWR + tRP
			t_refresh = 0

		if j == m:
			t_sum = t_sum + t_total - t_rest
			t_rest = t_total
	return(t_sum)