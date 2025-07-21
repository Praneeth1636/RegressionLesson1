import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt

data = fetch_california_housing()
X = pd.DataFrame(data.data , columns=data.feature_names)
Y = pd.Series(data.target)
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test , Y_pred)
r2 = r2_score(Y_test , Y_pred)
print("Mean Squared Error : " ,mse)
print("R Squ Score : " ,r2)
plt.scatter(X_test['MedInc'] , Y_test , color = 'blue' , label = 'Actual')
plt.scatter(X_test["MedInc"] , Y_pred , color="red",alpha=0.5,label='Predicted')
plt.xlabel('Median Inc')
plt.ylabel('House Price')
plt.legend()
plt.title('Cali House Pred')
plt.show()



