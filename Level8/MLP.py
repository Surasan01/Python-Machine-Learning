from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# โหลดข้อมูล MNIST จากไฟล์ .mat
mnist_raw = loadmat("mnist-original.mat")

# จัดเตรียมข้อมูลโดยแปลงเป็น dictionary ที่มี keys "data" และ "target"
mnist = {
    "data": mnist_raw["data"].T,  # เปลี่ยนจากคอลัมน์มาเป็นแถว
    "target": mnist_raw["label"][0]  # แปลง label เป็นแถวเดียว
}

# แยกข้อมูลเป็นตัวแปร x (ข้อมูลภาพ) และ y (ข้อมูล label)
x, y = mnist["data"], mnist["target"]

# สุ่มลำดับข้อมูล
shuffle = np.random.permutation(70000)
x, y = x[shuffle], y[shuffle]

# แยกข้อมูลออกเป็นชุดฝึก (training set) และชุดทดสอบ (test set)
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# แสดงขนาดของข้อมูลชุดฝึกและชุดทดสอบ
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# # (โค้ดที่ถูกคอมเมนต์) สร้างภาพแสดงข้อมูลภาพก่อนการฝึกโมเดล

# fig, ax = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[],'yticks':[]}, gridspec_kw=dict(hspace=0.1,wspace=0.1))

# for i, axi in enumerate(ax.flat):
#     axi.imshow(x_train[i].reshape(28,28), cmap='binary', interpolation='nearest')  # แสดงภาพที่ฝึก
#     axi.text(0.05, 0.05, str(int(y_train[i])), transform=axi.transAxes, color='black')  # แสดงเลข label ของภาพแต่ละภาพ

# plt.show()

# สร้างโมเดล Neural Network แบบ MLP (Multi-layer Perceptron)
model = MLPClassifier()
model.fit(x_train, y_train)  # ฝึกโมเดลด้วยข้อมูลชุดฝึก

# ทำนายผลด้วยข้อมูลชุดทดสอบ
y_pred = model.predict(x_test)

# แสดงความแม่นยำของโมเดลที่ทำนาย
print("Accuracy = ", accuracy_score(y_test, y_pred) * 100)

# สร้างภาพแสดงผลการทำนายภาพหลังจากฝึกโมเดลแล้ว
fig, ax = plt.subplots(10, 10, figsize=(8,8), subplot_kw={'xticks':[],'yticks':[]}, gridspec_kw=dict(hspace=0.1,wspace=0.1))

for i, axi in enumerate(ax.flat):
    # แสดงข้อมูลภาพในชุดทดสอบ
    axi.imshow(x_test[i].reshape(28,28), cmap='binary', interpolation='nearest')
    # แสดง label จริงของภาพในชุดทดสอบ
    axi.text(0.05, 0.05, str(int(y_test[i])), transform=axi.transAxes, color='black')
    # แสดงผลลัพธ์ที่โมเดลทำนาย (สีเขียวถ้าถูกต้อง สีแดงถ้าผิด)
    axi.text(0.75, 0.05, str(int(y_pred[i])), transform=axi.transAxes, color='green' if y_pred[i] == y_test[i] else "red")

# แสดงภาพ
plt.show()
