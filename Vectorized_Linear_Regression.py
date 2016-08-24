from numpy import *
import numpy as np

def total_error(c, m, points):
    totalErr=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalErr+=((m*x+c)-y)**2
    return float(totalErr)/(len(points))

def matrixify(X,y,points):
	X=[]
	a=np.array(points[:,0])
	
	for el in a:
		l=[1,el]
		X.append(l)

	X=np.matrix(X)
	y=[]
	a=np.array(points[:,1])
	for el in a:
		l=[el]
		y.append(l)
	return [X,y]
	



def sigma_cost(points,theta,X,y):
	cost1,cost2=0,0
	n = float(len(points))
	

	c1=np.matrix(np.zeros((2,1)))
	

	for i in range(100):
		# c1=c1+c1
		c1=c1+np.dot(np.sum(np.subtract(np.dot(X[i,:],theta),y[i])),X[i,:].T)

	return (0.0001/n)*c1
	


def gradient_descent_runner(points,alpha,m,c,total_iterations):

	theta=np.matrix(np.zeros((2,1)))
	X=[]
	y=[]
	X,y=matrixify(X,y,points)
	# y=np.matrix(y)	
	n = float(len(points))
	for i in range(total_iterations):
		theta=theta-2*sigma_cost(points,theta,X,y)	
	
	return theta
		
		

def run():
	points=genfromtxt("data.csv",delimiter=",")
	alpha=0.0001
	initial_c=0                             # initial guess of y-intercept
	initial_m=0							    # initial guess of slope

	total_iterations=1000  # total number of iterations  

	theta=gradient_descent_runner(points,alpha,initial_m,initial_c,total_iterations)
	print "theta[0] is ",(theta[0])
	print "theta[1] is ",(theta[1,:])
	print "net error with this hypothesis is ",total_error(theta[0], theta[1], points)   

if __name__ == '__main__':
    run()



