import pandas as pd
data = pd.read_csv('iris.csv')
# print(data)
# print(data.head())
# print(data.describe())     #It gives the properties of the data....z
# print(data.values)   #It gives daata in a matrix form...
# print(data.sort_values('SL',ascending = True))      #It arrange the data in the Ascending order based on the SL coloumn ...
# print(data.sample(10))  #It will give 10 random rows
# print(data.nlargest(5,'SL'))   #It will give the coloumn SL having the 5 large values
# print(data.nsmallest(5,'SL'))   #It will give the coloumn SL having the 5 small values
# print(data[data.SL > 7])  #It will give the rows having the SL value greater than 7
# print(data['SW'])         #It will give the coloumn SW

# print(data.loc[1:10 , 'SL':'PW'])  #It will give the rows from 1 to 10 and coloumns from SL to PW

# print(data.loc[data['SL'] > 5 , ['SW','SL']])  #It will give the rows having the SL value greater than 5 and the coloumns SL and SW
# print(data.iloc[2:5])print(data.loc[2])
print(data.sum('SW')) #It will give the sum of the coloumn SW