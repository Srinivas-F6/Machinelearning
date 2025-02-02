import pandas as pd
data = pd.read_csv(r"C:\Users\glaks\OneDrive\Desktop\Ml\datasets\car_evaluation.csv")

# preprocessing
columns = ["1","2","3","4","5","6","target"]

# Assinging coloums names to the data
data.columns = columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['1'] = le.fit_transform(data['1'])
data['2'] = le.fit_transform(data['2'])
data['3'] = le.fit_transform(data['3'])
data['4'] = le.fit_transform(data['4'])
data['5'] = le.fit_transform(data['5'])
data['6'] = le.fit_transform(data['6'])

# x = data.iloc[:,:-1]  
x=data.drop("target",axis=1)
y = data.iloc[:,-1]   # y = data['targer']

# model_selection
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.25,random_state = 1)

# model_Evaluation
from sklearn.tree import DecisionTreeClassifier

Rf = DecisionTreeClassifier()
Rf.fit(xtrain,ytrain)

ypred = Rf.predict(xtest)

from sklearn.metrics import accuracy_score
print(accuracy_score(ypred,ytest))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plot_tree(Rf,feature_names = x.columns, filled = True,rounded = True)
plt.show()


