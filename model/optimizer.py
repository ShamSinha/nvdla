#!/usr/local/bin/python3
'''Newton's Method Implementation'''
def function(x, a , b, c):
	''' this the funtion we want to find the roots of '''
	return (a*x*x + b*x + c)

def derivative_function(x, a, b):
	''' in this case we happen to know the differential of function '''
	''' can we also find the differential numerically '''
	return (2*a*x + b)

''' newton's method'''	
def newtons_method(guess):
	''' given a function find the root '''
	
	''' let's see if we can find these values numerically '''
	a = 1
	b = -3
	c = 2
	''' first, let's define a starting guess for x i.e. the root '''
	''' also , why does Newton's method work  '''
	''' come back to the question later '''
	n = 100
	x_guess =  guess # this is the starting guess that will initiate the process. BTW, what is a good guess anyway ?
	try:
		while(n != 0):
			
			f_at_x_guess = function(x_guess, a, b, c)
			del_f_at_x_guess = derivative_function(x_guess, a, b)
			x_guess = x_guess - f_at_x_guess/del_f_at_x_guess   # new guess for x based on the newton's method
			if(f_at_x_guess == 0.0):
				print("x = " , x_guess, "makes f(x) =  ", f_at_x_guess)
			n -= 1
	except ZeroDivisionError:
		print("Zero divison not possible...")

def construct_polynomial():
	''' given certain parameter, construct the function and return it '''
	pass
	
''' Now to find the derivative of a function numerically '''
def find_derivative():
	''' given a function find it's derivative '''
	pass

def perceptron():
	''' implement a simple perceptron '''


if __name__ == '__main__':
	guess = 0.00000000001
	newtons_method(guess)