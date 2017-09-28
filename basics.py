# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 14:03:09 2017

@author: shreya
"""
condition=1
while condition<10:
    print(condition)
    condition+=1

#infinite loop
#while True:
#    print('doing stuff!')
    
#string is given preference over int
#condition='2'
#while condition>5:
#    print ('test')

#for loop
exampleList=[1,5,6,6,1,5,2,1]
for x in exampleList:
    print(x)

#for loop using generator function, more efficient
for x in range(1,11):
    print(x)
    
#if-else, if-elif
x = 5
y = 8
if x > 55:
    print('x is greater than 55')
else:
    print('x is not less than 55')

#once a condition is satisfied the loop is broken , no further if's are checked
x = 5
y = 10
z = 22
if x < y:
    print('x is greater than y')
elif x < z:
    print('x is less than z')
else:
    print('if and elif never ran...')
    
    
 #multiple if run and the last else is related to last if   
x = 5
y = 10
z = 22
if x < y:
    print('x is greater than y')
if x < z:
    print('x is less than z')
else:
    print('if and elif never ran...')
    
#defining functions
def example():
    print('this code will run')
    z=3+5
    print(z)
    
example()

def add_numbers(x, y, z=None, flag=False):
    if (flag):
        print('Flag is true!')
    if (z==None):
        return x + y
    else:
        return x + y + z
    
print(add_numbers(1, 2, flag=True))

#local and global variables
x = 6

def example():
    print(x)
    # z, however, is a local variable.  
    z = 5
    # this works
    print(z)
    
example()

x = 6
def example2():
    # works
    print(x)
    print(x+5)

    # but then what happens when we go to modify:
    x+=6
    
x = 6
def example3():
    # what we do here is defined x as a global variable. 
    global x
    # now we can:
    print(x)
    x+=5
    print(x)
    
x = 6
def example4():
    globx = x
    # now we can:
    print(globx)
    globx+=5
    print(globx)

x = 6
def example(modify):
    print(modify)
    modify+=5
    print(modify)
    return modify
 
x = example(x)
print(x)    
    
#writing to a file
text = 'Sample Text to Save\nNew line!'    
saveFile = open('exampleFile.txt','w')
saveFile.write(text)
saveFile.close()
#appending in a file
appendMe = '\nNew bit of information'
appendFile = open('exampleFile.txt','a')
appendFile.write(appendMe)
appendFile.close()
#read from a file
readMe = open('exampleFile.txt','r').read()
print(readMe)   
#read line by line
readMe = open('exampleFile.txt','r').readlines()
print(readMe) 

if __name__ == '__main__':
    print('such great module!!!!')
    
''' multi line comment '''

#getting user input
x= input('what is your name?:')

#importing modules
import statistics
import statistics as s
from statistics import mean
from statistics import mean as m, median as d
from statistics import *

#tuples and lists
x=(1,'a',3,'b') #tuple is immutable
x=[1,'a',2,'b'] #list is mutable


#manipulating lists
[1,2]+[3,4] #concatenation
[1,3]*3  #repetition of list
x=[1,2,3,4,5,6]
x.append(7)
x.insert(2,33)
x.remove(6)
print(x[5])
print(x.index(1)) #returns index of value 1
print(x.count(1))
x.sort() #in case of string, the data gets alphabetically sorted

#in operator
1 in[1,2,3]
name= 'Christopher'
'Chris' in name

#multidimensional Lists
x=[[2,6],[6,3],[4,5],[5,12]]
print(x[2])
print(x[2][1])

#strings
x='this is a string'
print(x[0])
print(x[0:1])
print(x[0:2])
x[-1]
x[-4:-2]
x[:3]

firstname = 'Christopher Arthur Hansen Brooks'.split(' ')[0]


##reading a csv- a csv is read as dictionary in pyhton with column names as keys
#import csv
#with open('basic.csv') as csvfile:
#    readCSV=list(csv.reader(csvfile,delimiter=','))
#    for row in readCSV:
#        print(row)
#        print(row[0])
#        print(row[0],row[1],row[2])
#    dates=[]
#    colors=[]
#    
#    for row1 in readCSV:
#        color=row1[3]
#        date=row1[0]
#        dates.append(date)
#        colors.append(color)
#    print(dates)
#    print(colors)
#        
#dictionaries
x={'christopher':23,'james':25,'harry':34}
x['christopher']

#for name in x:   #using index
#    print(x[name])
#   
#for age in x.values:   #using values
#    print(age)

for name,age in x.items():
     print(name)
     print(age)
    
    

#types and sequences
type('this is a string')
type(None)
type(1)
type(1.0)
type(add_numbers)

#unpacking a sequence
x=('christopher','Brooks',23)
fname, lname, age=x

#format operator
sales_record={'price' : 3.24,'num_items': 4,'person':'Chris'}
print('{} bought {} items at a price of {} each for a total of {}'.format(sales_record['person'],sales_record['num_items'],sales_record['price'],sales_record['num_items']*sales_record['price']))

#set: returns unique values in the list
list1=[1,1,2,2,3,3,4,5,6,7,8,8]
unique_list1=set(list1)

#dates and time
import datetime as dt
import time as tm
tm.time()
dtnow=dt.datetime.fromtimestamp(tm.time())
dtnow.year, dtnow.month, dtnow.day
delta=dt.timedelta(days=100)
today=dt.date.today()
today-delta
tm.sleep(2)    #time delay of 2 seconds

#class in python
class person:
    department='IIT'
    def set_name(self,new_name):
        self.name=new_name
    def set_locatiion(self,new_location):
        self.location=new_location
    
p1= person()
p1.set_name('Shreya Sharma')
p1.set_locatiion('Roorkee, india')
print('{} live in{} and studied in the department of{}'.format(p1.name,p1.location,p1.department))

#map: faster than iterative for loop
store1 = [10.00, 11.00, 12.34, 2.34]
store2 = [9.00, 11.10, 12.34, 2.01]
cheapest =list( map(min, store1, store2))

x=tuple(map(int,"3 4".split(' ')))
print(x)

#Lambda, map, reduce, filter
f= lambda a,b:a*b
print(f(2,3)) 

def add10(z):
    z+=10
    return z
    
def add20(z):
    z+=20
    return z
  
def addition(a,b):
    x=a+b
    return x

def subtraction(a,b):
    x=a-b
    return x
 
a=[1,2,3,4,5]
b=[3,5,6,8,9]

#map with one variable , map() is faster than the iterative for() operation
c= list(map(add10,a))

#map with two variables
d= list(map(addition,a,b))
d=list(map(subtraction,a,b))

#map using lambda, due to lambda , no need to define function earlier 
#lambda is a replacemnt of define function, map is replacement of for loop
d= list(map(lambda x:x+10, a))
d= list(map(lambda x,y:x+y, a,b))

#map with  multiple functions
funcs= [add10, add20]
for  i in range(10):
    d= list(map(lambda x:x(i), funcs)) 

# filter
x= range(0,10)
y= list(filter(lambda x:x%2==0,x))
y= list(filter(lambda x:x%2,x))

#reduce
from functools import reduce
x= range(1,10)
y= reduce(lambda a,b: a*b,x)
print(y)  
f = lambda a,b: a if (a > b) else b
print(reduce(f, [47,11,42,102,13]))

#list comprehension
my_list=[x for x in range(0,1000) if x%2==0]
z=[x**3 for x in range(10)]

#set comprehension
squared={x**2 for x in range(1,10)}  
print(squared)
























 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    