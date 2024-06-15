from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris_dataset = load_iris()
# print(iris_dataset.data.shape)
# (150,4)
# 75% , 25%

x_train,x_test,y_train,y_test = train_test_split(iris_dataset["data"],iris_dataset["target"],test_size=0.2,random_state=0)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# 150
# train 75% = 112
# test 25% = 38

# test_size !! is test 20% = 30 , train 80% = 120