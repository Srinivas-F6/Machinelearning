import pandas as pd
data = pd.read_csv(r"C:\Users\glaks\OneDrive\Desktop\Ml\datasets\iris.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



x = data.iloc[:,:-1]
y = data.iloc[:,-1]

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 1)

from sklearn.neighbors import RadiusNeighborsClassifier

rn = RadiusNeighborsClassifier(radius = 2.5)

rn.fit(xtrain,ytrain)

ypred = rn.predict(xtest)

print(ypred)
from sklearn.metrics import accuracy_score

print(accuracy_score(ytest,ypred))