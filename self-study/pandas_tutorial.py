# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:39:49 2017

@author: hoge
"""

import pandas as pd
import numpy as np

#creating series
animals=['Tiger','Bear','moose']
pd.Series(animals)
numbers=[1,2,3]
pd.Series(numbers)
numbers=[1,2,None]
pd.Series(numbers)
np.nan==None #important difference between Nan and None

np.isnan(np.nan)
sports={'archery':'bhutan','Golf':'scotland','Sumo': 'Japan','Taekwondo': 'South Korea'}
s=pd.Series(sports)
idx=s.index
s= pd.Series(['Tiger', 'Bear', 'Moose'], index=['India', 'America', 'Canada'])
s= pd.Series(sports,index=['Golf','Sumo','Hockey'])

#Querying a series
s=pd.Series(sports)
a=s.iloc[3]
a=s.loc['Golf']
sports = {99: 'Bhutan',
          100: 'Scotland',
          101: 'Japan',
          102: 'South Korea'}
s = pd.Series(sports)
#s[0] won't work, s.iloc[0] works

s=pd.Series([120.00,120.0,101.00,3.00])
total=0
for item in s:
    total+=item
#print(total)

total=np.sum(s)

#this creates a big series of random numbers
s=pd.Series(np.random.randint(0,1000,10000))
print(s.head())
print(len(s))

"""%%timeit -n 100
summary=0
for item in s:
    summary+=item
    # to time any loop(np inbuilt commands such as sum are faster than for loops)"""
#Broadcasting
s+=2
print(s.head())

for label, value in s.iteritems():#iteritems() is used to iterate over key-value pairs in dictionary; It's basically a difference in how they are computed. items() creates the items all at once and returns a list. iteritems() returns a generator--a generator is an object that "creates" one item at a time every time next() is called on it
    s.set_value(label,value+2)
print(s.head())
    
#%%timeit -n 10
#s = pd.Series(np.random.randint(0,1000,10000))
#for label, value in s.iteritems():
#    s.loc[label]= value+2

#%%timeit -n 10
#s = pd.Series(np.random.randint(0,1000,10000))
#s+=2
 
s= pd.Series([1,2,3])
s.loc['Animal']='Bears' #a series can have multple data types

original_sports = pd.Series({'Archery': 'Bhutan',
                             'Golf': 'Scotland',
                             'Sumo': 'Japan',
                             'Taekwondo': 'South Korea'})
#all the indices must be defined even if the index is same for all
cricket_loving_countries = pd.Series(['Australia',
                                      'Barbados',
                                      'Pakistan',
                                      'England'], 
                                   index=['Cricket',
                                          'Cricket',
                                          'Cricket',
                                          'Cricket'])
all_countries = original_sports.append(cricket_loving_countries)
print(all_countries)
print(all_countries.loc['Cricket'])

#Creating a dataframe
purchase_1 = pd.Series({'Name': 'Chris',
                        'Item Purchased': 'Dog Food',
                        'Cost': 22.50})
purchase_2 = pd.Series({'Name': 'Kevyn',
                        'Item Purchased': 'Kitty Litter',
                        'Cost': 2.50})
purchase_3 = pd.Series({'Name': 'Vinod',
                        'Item Purchased': 'Bird Seed',
                        'Cost': 5.00})
df = pd.DataFrame([purchase_1, purchase_2, purchase_3], index=['Store 1', 'Store 1', 'Store 2'])
print(df.head())
a=df.loc['Store 1']
type(a) #a datframr as multiple rows have same index
a=df.loc['Store 2'] #loc is used to access a row
type(a)
a=df.loc['Store 1','Cost']
print(df.T)
print(df.T.loc['Cost'])
print(df['Cost']) #to access a column
df.loc['Store 1']['Cost']
df.loc[:,['Name', 'Cost']]
df.drop('Store 1') #the original dataframe remains unchanged

#Creating a copy of Dataframe
copy_df = df.copy()
copy_df = copy_df.drop('Store 1')
print(copy_df)
del copy_df['Name']
print(copy_df)
df['Location']=None
print(df)

#Dataframe indexing and loading
costs= df['Cost']
print(costs)
costs+=2  #broadcasting in DF

import csv
#with open('data - Copy.csv') as csvfile:
#    readCSV=list(csv.reader(csvfile,delimiter=','))
#
#
#   
#with open('data.csv', 'w') as csvfile:
#    attributes=['symboling', 'normalized-losses','make', 'fuel', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location',  'wheel-base', 'length','width','height','curb-weight','engine-type','no. of cylinders','engine-size','fuel system','bore','stroke','compression-raio','horsepower','peak-rpm',
#            'city-mpg','highway-mpg','price']
#
#    writer = csv.DictWriter(csvfile, fieldnames=attributes)
    
    
df=pd.read_csv('data - Copy.csv')
#df=df.shift(periods=1, axis=0) #to shift by rows(axis=0) or by columns(axis=1) by periods in a dataframe
print(df.columns)
df['wheel-base']>100 #creating a boolean mask 
#A Boolean mask is an array which can be of one dimension like a series, or two dimensions like a DataFrame, where each of the values in the array are either true or false. This array is essentially overlaid on top of the data structure that we're querying. And any cell aligned with the true value will be admitted into our final result, and any sign aligned with a false value will not. 
masked_df1=df.where(df['wheel-base']>100) # in this method ,NAN values are present where the value is false
masked_df2= df[df['wheel-base']>100] #in this method, NAN are automatically removed by the pandas
masked_df1=masked_df1.dropna() #to remove NAN from in case of method 1

print(len(df[(df['wheel-base']>100)|(df['length']>150)])) #logical operators in boolean masking, take care of the brackets
print(len(df[(df['wheel-base']>100)& (df['length']>150)]))

#indexing Dataframes
df=df.set_index('make')
df=df.reset_index()
print(df['make'].unique())
df=df.set_index(['make','body-style'])  #multi-indexing
df1=df.loc['audi','sedan']
df2=df.loc[[('audi','sedan'),('audi','wagon')]] #take note of number of square brackets

#missing values
df['make-style']=df.index
df=df.set_index(['price','make-style'])
df=df.sort_index()

df=df.fillna(method='ffill') # When you use statistical functions on DataFrames, the functions typically ignore missing values.
#'?' is taken as some value and not NAN , therefore fillna() is not able to identify it as empty cell

#merging dataframes
df= pd.DataFrame([{'Name':'Chris','item purchased':'Sponge','Cost':22.50},{'Name':'Kevyn','item purchased':'Kitty Litter','Cost':2.50},{'Name':'Filip','item purchased':'Spoon','Cost':5.00}], index=['store1','store1','store2'])
df['date']=['december1','january1','mid-day']
df['delivered']=True
df['feedback']=['positive',None,'negative']
adf=df.reset_index()
adf['date']=pd.Series({0:'December 1',1:'mid-may'})

staff_df=pd.DataFrame([{'Name':'kelly','Role':'Director'},
                       {'Name':'sally','Role':'instructor'},
                       {'Name':'james','Role':'grader'}])

staff_df=staff_df.set_index('Name')
student_df= pd.DataFrame([{'Name':'james','school':'business'},
                          {'Name':'Mike','school':'law'},
                          {'Name':'sally','school':'engineering'}])
student_df=student_df.set_index('Name')

union_df=pd.merge(staff_df, student_df, how='outer', left_index=True, right_index=True)
intersection_df=pd.merge(staff_df, student_df, how='inner', left_index=True, right_index=True)
staff_all_df=pd.merge(staff_df, student_df, how='left', left_index=True, right_index=True)
student_all_df=pd.merge(staff_df, student_df, how='right', left_index=True, right_index=True)

# we can also use columns to jpin the dataframes
staff_df=staff_df.reset_index()
student_df= student_df.reset_index()
merge_df= pd.merge(staff_df, student_df , how= 'left', left_on='Name', right_on='Name')

#conflicts in column
staff_df=pd.DataFrame([{'Name':'kelly','Role':'Director', 'location':'X'},
                       {'Name':'sally','Role':'instructor','location':'X2'},
                       {'Name':'james','Role':'grader','location':'X2'}])

student_df=pd.DataFrame([{'Name':'james','school':'business','location':'X1'},
                          {'Name':'Mike','school':'law','location':'X1'},
                          {'Name':'sally','school':'engineering','location':'X1'}])

merge_df= pd.merge(staff_df, student_df , how= 'left', left_on='Name', right_on='Name')

#idiomatic expresions: more readable ansd more efficient code. The multiple line functions are compressed and consized in single line through idiomatic expressions. Some of the methods are chaining, map
