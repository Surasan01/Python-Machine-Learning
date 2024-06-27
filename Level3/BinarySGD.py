from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix # แสดงผลการทดลองและค่าจริง
from sklearn.model_selection import cross_val_predict

import itertools
from sklearn.metrics import classification_report # Precision , Recall , F1 , Accuracy
from sklearn.metrics import accuracy_score

mnist_raw = loadmat("mnist-original.mat")

def displayimage(x):
    plt.imshow(x.reshape(28,28),
               cmap=plt.cm.binary,
               interpolation="nearest")
    plt.show()

def displayPredict(clf,actually_y,x):
    print("Actually = ",actually_y)
    print("Prediction = ",clf.predict([x])[0]) # [0] เพื่อเอาตัวแรก

def displayConfusionMatrix(cm,cmap=plt.cm.GnBu):
    classes=["Other Number","Number 5"]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    trick_marks=np.arange(len(classes))
    plt.xticks(trick_marks,classes)
    plt.yticks(trick_marks,classes)
    thresh=cm.max()/2
    for i , j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],'d'),
        horizontalalignment='center',
        color='white' if cm[i,j]>thresh else 'black')

    plt.tight_layout()
    plt.ylabel('Actually')
    plt.xlabel('Prediction')
    plt.show()

mnist = {
    "data": mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}
# print(mnist)

# print(mnist["data"].shape)
# print(mnist["target"].shape)

x,y = mnist["data"],mnist["target"]
# training , test
# 1-60000
# 60001-70000
# training , test set
x_train,x_test,y_train,y_test = x[:60000],x[60000:],y[:60000],y[60000:]



# class 0-9

# class 0 , ไม่ใช่ class 0

# ข้อมูลค่า 5000 -> model -> class หรือไม่ ? true : false
# y_train = [0,0,....,9...,9]

predict_number = 5500

# y_train_0 = (y_train==0)
# y_test_0 = (y_test==0)

# class 5 , ไม่ใช่ class 5
y_train_5 = (y_train==5)
y_test_5 = (y_test==5)

# y_train_0 = [true,true,....,false...,false]
# print(y_train_0.shape,y_train_0)
# print(y_test_0.shape,y_test_0)

sgd_clf = SGDClassifier()
sgd_clf.fit(x_train,y_train_5)

# เช็ค
# displayimage(x_test[predict_number])
# displayPredict(sgd_clf,y_test_5[predict_number],x_test[predict_number])

score = cross_val_score(sgd_clf,x_train,y_train_5,cv=4,scoring="accuracy")
print(score)

y_train_pred = cross_val_predict(sgd_clf,x_train,y_train_5,cv=4)
cm = confusion_matrix(y_train_5,y_train_pred)
# print(cm)

# plt.figure()
# displayConfusionMatrix(cm)

# การทดลองข้อมูล
y_test_pred = sgd_clf.predict(x_test)

classes = ["Other Number","Number 5"]
print(classification_report(y_test_5,y_test_pred,target_names=classes))
print("Accuracy = ",accuracy_score(y_test_5,y_test_pred)*100)