from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score

iris_dataset = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris_dataset['data'],iris_dataset['target'],test_size=0.4,random_state=0)

print(x_train)

# Model
knn = KNeighborsClassifier(n_neighbors=5) # ดูตัวเปรียบเทียบกี่ตัว

# training
knn.fit(x_train,y_train)


# print(x_test[1])
# print(y_test[1])

# prediction

# pred = knn.predict([x_test[0]]) # ใส่ [] เพื่อให้เป็นอาเร 2 มิติ
# print("ผลการพยากรณ์ = ",pred)
# print("ทำนายว่าอยู่ในกลุ่มสายพันธ์ุ ",iris_dataset["target_names"][pred])

y_pred = knn.predict(x_test)

print(classification_report(y_test,y_pred,target_names=iris_dataset["target_names"]))
print(x_test.shape)
print("ความแม่นยำ = ",accuracy_score(y_test,y_pred)*100)