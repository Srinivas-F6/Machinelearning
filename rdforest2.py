import pandas as pd
data = pd.read_csv(r"C:\Users\glaks\OneDrive\Desktop\Ml\datasets\income_evaluation.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data[' workclass'] = le.fit_transform(data[' workclass'])
data[' education'] = le.fit_transform(data[' education'])
data[' marital-status'] = le.fit_transform(data[' marital-status'])
data[' occupation'] = le.fit_transform(data[' occupation'])
data[' relationship'] = le.fit_transform(data[' relationship'])
data[' race'] = le.fit_transform(data[' race'])
data[' sex'] = le.fit_transform(data[' sex'])
data[' native-country'] = le.fit_transform(data[' native-country'])

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.32, random_state = 1)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(xtrain,ytrain)

ypred = rf.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ypred,ytest))