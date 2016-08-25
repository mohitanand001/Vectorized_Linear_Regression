import numpy as np
# from random import *
import random
def cost_now(x,y,theta,numofpoints):
	hypo=np.dot(x,theta)
	err=hypo-y
	cost=(1.0/numofpoints)*(np.sum(err**2))	
	return cost




def gradient_descent(x,y,theta,alpha,numofpoints,total_iterations):
	xtr=x.transpose()
	for i in range(total_iterations):
		# we use dot operation so that there is matrix multiplication between x and theta
		# which are numpy arrays. simply using a '*' will not produce the desired results
		hypothesis=np.dot(x,theta)
		error=hypothesis-y		
		# cost= np.sum(error**2)*(1.0/numofpoints)
		partial_derivative_cost=np.dot(xtr,error)
		theta=theta-(alpha/numofpoints)*partial_derivative_cost

	return theta	



def random_data_genr(numofpoints,bias,variance):
	x=np.zeros(shape=(numofpoints,2))
	y=np.zeros(shape=numofpoints)
	# theta=np.zeros(shape=(2),)

	for i in range(numofpoints):
		x[i][0]=1
		x[i][1]=i
		y[i]=(i+bias)+random.uniform(0,1)*variance

	return x,y


def main():
	numofpoints=100
	alpha=0.0001
	total_iterations=10000
	bias=25
	variance=10
	x,y=random_data_genr(numofpoints,bias,variance)
	theta=np.zeros(shape=(2,))
	theta=gradient_descent(x,y,theta,alpha,numofpoints,total_iterations)
	print theta
	print cost_now(x,y,theta,numofpoints)



if __name__=="__main__":
	main()




