import pandas as pd
data = pd.read_csv(r"C:\Users\glaks\OneDrive\Desktop\Ml\datasets\dataset_traffic_accident_prediction1.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['Weather'] = le.fit_transform(data['Weather'])
data['Weather'] = data['Weather'].fillna(data['Weather'].mean())


data['Time_of_Day'] = le.fit_transform(data['Time_of_Day'])
data['Time_of_Day'] = data['Time_of_Day'].fillna(data['Time_of_Day'].mean())

print(data)