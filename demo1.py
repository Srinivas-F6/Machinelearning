import pandas as pd
data = pd.read_csv("C:\\Users\\glaks\\OneDrive\\Desktop\\Ml\\iris.csv")
# print(data['SL'])
# dataframe is a datatype in pandas
x = data.iloc[:,:-1].values    #here .values is converted data into matrices......
y = data.iloc[101:150,-2]
print(x)
print(y)


# from sklearn.model_selection import train_test_split


from sklearn.neighbors import KNeighborsClassifier  #
model = KNeighborsClassifier(n_neighbors = 3)

model.fit(x_train,y_train)


