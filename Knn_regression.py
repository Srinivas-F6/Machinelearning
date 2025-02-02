import pandas as pd
data = pd.read_csv('C:\\Users\\glaks\\OneDrive\\Desktop\\Ml\\Student_Performance.csv')
# print(data.head())

# independent values
x = data.iloc[:,:-1].values

# dependent values
y = data.iloc[:,-1]

# train_test_split --> this splits the data into training and testing 

from sklearn.model_selection import train_test_split
# test_size = 0.2 --indicates 20% of the go for the testing.

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)

from sklearn.neighbors import KNeighborsRegressor

model = KNeighborsRegressor(n_neighbors = 5)

model.fit(x_train,y_train)

ypred = model.predict(x_test)

print(ypred)

