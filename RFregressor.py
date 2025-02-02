import pandas as pd
data = pd.read_csv(r"C:\Users\glaks\OneDrive\Desktop\Ml\datasets\creditcard.csv")


from sklearn.preprocessing import LabelEncoder
 
le = LabelEncoder()

data = data.apply(le.fit_transform)

x = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.32, random_state = 1)

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor()

rf.fit(xtrain,ytrain)

ypred = rf.predict(xtest)

from sklearn.metrics import r2_score
print(accuracy_score(ypred,ytest))
