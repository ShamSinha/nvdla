//SRAM time calculation method
import math
bit = input ('#bits : ')
bi = int(bit) - 512
zheng = bi//2048
yu = bi%2048
number_512 = math.ceil (yu/512)
t = 2 + zheng*5 + number_512*2
print(t)