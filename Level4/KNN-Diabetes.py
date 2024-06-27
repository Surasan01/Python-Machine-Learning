from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np # เอาไว้แปลเป็น 2 มิติ
import pandas as pd # เอาไว้อ่านไฟล์
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix

df=pd.read_csv("diabetes.csv")

# print(df.head())
# print(df.shape)

#data
x = df.drop("Outcome",axis=1).values # values ทำให้เป็น array 2 มิติ
# outcome data
y = df['Outcome'].values
# print(x)
# print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)

# find k to model 1-8
k_neighbors = np.arange(1,9)
# [1,2,3,4,5,6,7,8]

# empty
training_score = np.empty(len(k_neighbors))
test_score = np.empty(len(k_neighbors))



for i,k in enumerate(k_neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    training_score[i]=knn.score(x_train,y_train)
    test_score[i]=knn.score(x_test,y_test)
    print(test_score[i]*100)
    print(training_score[i]*100)

# plt.title("Compare K value in model")
# plt.plot(k_neighbors,test_score,label="Test Score")
# plt.plot(k_neighbors,training_score,label="Train Score")
# plt.xlabel("K Number")
# plt.ylabel("Score")
# plt.show()

# 7 แม่นสุด
knn = KNeighborsClassifier(n_neighbors=8)
# training
knn.fit(x_train,y_train)

# prediction
y_pred = knn.predict(x_test)

# print(classification_report(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred ))

# แสดงความผิดพลาด กี่ตัว
print(pd.crosstab(y_test,y_pred,rownames=["True"],colnames=["Prediction"],margins=True)) 
