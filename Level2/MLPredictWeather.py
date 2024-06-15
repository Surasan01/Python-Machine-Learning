import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv("https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv")

# print(dataset.shape)
# print(dataset.describe())
# dataset.plot(x='MinTemp',y='MaxTemp',style='o')
# plt.title("Min & Max Temp")
# plt.xlabel("Mintemp")
# plt.ylabel("Maxtemp")
# plt.show()

# train & test set
x = dataset["MinTemp"].values.reshape(-1,1)
y = dataset["MaxTemp"].values.reshape(-1,1)

# 80% - 20%
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=0)

# training
model = LinearRegression()
model.fit(x_train,y_train)

# test model
y_pred = model.predict(x_test)


# compare true data & predict data
df = pd.DataFrame({'Actually':y_test.flatten(),'Predicted':y_pred.flatten()})

df1 = df.head(20)
df1.plot(kind="bar",figsize=(16,7))
plt.show()

# plt.scatter(x_test,y_test)
# plt.plot(x_test,y_pred,color="red",linewidth=2)
# plt.show()

print("MAE = ",metrics.mean_absolute_error(y_test,y_pred))
print("MSE = ",metrics.mean_squared_error(y_test,y_pred))
print("RMSE = ",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print("Score = ",metrics.r2_score(y_test,y_pred))
