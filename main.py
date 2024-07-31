import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data=pd.read_csv('/content/House Price India.csv')

data['Total area']=data['Area of the basement']+data['Area of the house(excluding basement)']

x=data[['number of bedrooms','number of bathrooms','Total area']]
y=data.drop('Price',axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model=LinearRegression();

model.fit(X_train,y_train);

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')
