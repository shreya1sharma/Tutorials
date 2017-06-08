# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:44:20 2017

@author: 0000016446351
"""
import numpy as np

#creating numpy arrays
my_list=[1,2,3]
x= np.array(my_list)
y=np.array([4,5,6])

#creating multi-dimensional array: by passing list of lists
m= np.array([[1,2,3],[1,4,8]])
m.shape

#arange returns evenly spaced values within a given interval
n= np.arange(0,30,2)
n=n.reshape(3,5)
n=np.linspace(0,4,9)
#n=n.resize(3,3)
np.ones((3,2))
np.zeros((2,3))
np.eye(3)
np.diag(x)

#creating array of repeating elements
n=np.array([1,2,3]*3)
n=np.repeat([1,2,3],3)

#combining arrays
p= np.ones([2,3],int)
np.vstack([p,2*p]) #row wise
np.hstack([p,2*p]) #column wise

#operations
print(x+y)
print(x-y)
print(x*y)
print(x/y)
print(x**2)
z1=np.dot(x,y)
z=np.array([y,y**2])
len(z) #gives the number of rows
z=z.T #transpose
z.dtype
z=z.astype('f')   #cast to a specific type

#math functions
a=np.array([-4,-2,1,3,4])
a.sum()
a.max()
a.min()
a.mean()
a.std()
a.argmax()  #gives the index of maximum value in array
a.argmin()

#indexing/slicing
s=np.arange(13)**2
s[4]
s[0]
s[1:5] #shows the range array[start:stop]
s[-4:]
s[-5::-2] #shows range with step size array[start:stop:stepsize]

r=np.arange(36)
r.resize((6,6))
r[2,2]  #for multi-dimesional array
r[3, 3:6]
r[:2,:-1]
r[-1,::2]
r[r>30]  #conditional indexing
r[r>30]=30

#copying data
r2=r[:3,:3]
r2[:]=0 # r also changes :(
#to avoid this
r_copy=r.copy() 
r_copy[:]=10 #r remains same

#iterating over arrays
test= np.random.randint(0,10,(4,3))
for row in test:    #iterate by row
    print(row)
  
for i in range(len(test)):   #iterate by index
    print(test[i])

for i,row in enumerate(test):
    print('row',i,'is',row)
    
#zip
test2=test**2

for i,j in zip(test,test2):
    print(i,'+',j,'=',i+j)





































