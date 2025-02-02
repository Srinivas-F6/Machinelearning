import pandas as pd
data = pd.read_csv(r"C:\Users\glaks\OneDrive\Desktop\Ml\datasets\excell.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['clock_speed'] = le.fit_transform(data['clock_speed'])
data['m_dep'] = le.fit_transform(data['m_dep'])

x = data.iloc[:,:-1]
y = data.iloc[:,-1]


from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size = 0.7,random_state = 1)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()

model.fit(xtrain,ytrain)

ypred = model.predict(xtest)

from sklearn.metrics import r2_score

print(r2_score(ypred,ytest))
