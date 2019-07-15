'''
how many cycles the neural network will take 
'''

import numpy as np
import pandas as pd

#gobal vars
ATOMIC_OPERATIONS = 1     #already have the number of atomic operations in default

#the formula that we use : total_time=Beginning_time + Common_part + Ending_time
#Beginning_time for the time that the data fills the CBUF. 
#Common_part is the CMAC running time
#Ending_time is the truncationg --> DBUF --> SDP --> SRAM part
#unit: cycles
'''
865 is channel operations. 
But for different nn, the number of atomic operations in channel operation will change.
it is better to calculate the cycles based on atomic operation
'''
def Total_Cycles(atomic_operations):
    Beginning_time = 6469
    #Common_part = 865 * channel_operations
    Common_part = 16 * atomic_operations
    Ending_time = 26
    total_time =Beginning_time + Common_part + Ending_time
    return total_cycles

'''
the information of all layers should first be read and saved in the lists
Cause the space, comments, '\n', all these have been recorded in contents list. We should do more to this lists to make sure that the information in the list is useful.
'''
def load_file():
    a_flag = 0 
    i_flag = 0                                   #these two flags are used for delet the comments in the beginning of txt file
    b_flag = 0                                   #this flag is for deleting the last comment line in the list
    contents = []
    #firstly, read layers information from the existed file
    with open('./nvdla_input.txt', 'r') as file_to_read:
        while True:
            line = file_to_read.readline()       #read one line, and store as a list
            if not line:
                break
            line = line.strip('\n')              #remove the '\n' in the end
            if line[:3]=='\'\'\'':               #detect the first three character in this line. If this is a comment, we wont add it in the list.
                i_flag = 1
                if a_flag == 1:
                    i_flag = 0
            a_flag = i_flag
            if i_flag != 1:                      #we wont add the comments lines into the final list
                if b_flag == 0:                  #remove the last line of the comments
                    b_flag = 1
                else:
                    line = line.split(' ')       #convert the line string in the list into the different strings, based on the character:' 'space
                    label_list = [int(i) for i in line]  #convert the string type to int type
                    contents.append(label_list)
    return contents

'''
calculate how many atomic_operations it will have
feature_size = [h, w, c]
weight_size = [h, w, c]       
'''
def Number_Atomic_0p():
    contents = load_file()
    #layer =np.vstack((feature_size, weight_size, stripe, padding))                       #https://blog.csdn.net/daoxiaxingcai46/article/details/78269910
    i = 1
    total_computations = 0
    for line in contents:
        if i == 1:
            i = i + 1
            feature_map = line   #get the image from the first line in the file
        else:
            #calculate the output size
            output_size = (feature_map[0] + 2*line[4] - line[0])//line[3] +1         #using // instead /
            #how many computations in one layer it will take
            computations_in_surface = output_size * line[0] * line[1]                #one output means one kernel computation. One kernel computation means w*h comptations
            computations_in_layer = total_computation_in_surface * line[2] * (feature_map[2]+63)//64 #in the channel direction, how many times it will cycle. "+63" will keep the output of "/64" to the maximum integer
            #update the feature_map
            feature_map[0] = output_size  #output feature map width
            feature_map[1] = output_size  #output feature map height
            feature_map[2] = line[3]      #output feature map channel
            #update total computations numbers
            total_computations = total_computations + computations_in_layer 
    return total_computations

'''
in the main function
we will first get the number of atomic operations 
then get the total cycles based on atomic operations
'''	
def main():
    if ATOMIC_OPERATIONS:                               #already have the number of atomic operations
        total_cycles = Total_Cycles(A_o) 
    else:                                               #based on layers
        total_atomic_op = Number_Atomic_0p()            #首先定义layer 得到atomic operationss
        total_cycles = Total_Cycles(total_atomic_op)    #再其次调用Total_Cycles
    print("Total cycles for this neural network is :")
    print(total_cycles)

'''
execution lines
'''
ATOMIC_OPERATIONS = 0        #change this value if you already have the total atomic_operations
#ATOMIC_OPERATIONS = 1
#A_o = 35586                 #A_o is the total atomic_operations
main()