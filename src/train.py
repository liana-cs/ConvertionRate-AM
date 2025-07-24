import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('../data/KAG_conversion_data.csv')

x = df.drop(columns=['Total_Conversion'])
y = df['Total_Conversion']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)


print("Média do conjunto total:", y.mean())
print("Média do treino:", y_train.mean())
print("Média do teste:", y_test.mean())