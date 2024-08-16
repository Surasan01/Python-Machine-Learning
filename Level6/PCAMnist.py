from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt

# โหลดข้อมูล MNIST จากไฟล์ .mat
mnist_raw = loadmat("mnist-original.mat")

# แปลงข้อมูล MNIST ให้อยู่ในรูปแบบที่สามารถใช้งานได้ง่าย
mnist = {
    "data": mnist_raw["data"].T,  # Transpose data เพื่อให้แถวเป็นตัวอย่างและคอลัมน์เป็นคุณลักษณะ
    "target": mnist_raw["label"][0]  # ดึงข้อมูลป้ายกำกับ (labels)
}

# แบ่งข้อมูลออกเป็นชุดการฝึก (training set) และชุดทดสอบ (test set)
x_train, x_test, y_train, y_test = train_test_split(mnist["data"], mnist["target"], random_state=0)

# สร้างโมเดล PCA ที่เก็บ 80% ของความแปรปรวน (variance)
pca = PCA(.8)
data = pca.fit_transform(x_train)  # ลดมิติข้อมูลการฝึก
result = pca.inverse_transform(data)  # แปลงข้อมูลกลับไปเป็นมิติเดิมเพื่อให้เปรียบเทียบได้

print(x_train.shape)
# แสดงผลลัพธ์ของข้อมูลหลังจากการลดมิติ
print(data.shape)

# แสดงภาพต้นฉบับและภาพหลังจากการลดมิติและแปลงกลับ
plt.figure(figsize=(8, 4))

# ภาพแรก: แสดงภาพต้นฉบับจากข้อมูล MNIST ที่มี 784 คุณลักษณะ
plt.subplot(1, 2, 1)
plt.imshow(mnist["data"][0].reshape(28, 28), cmap=plt.cm.gray, interpolation="nearest")
plt.xlabel("784 Features")  # บอกผู้ใช้ว่าภาพนี้ใช้คุณลักษณะ 784 จุดในการสร้าง

# ภาพที่สอง: แสดงภาพหลังจากการลดมิติข้อมูลด้วย PCA และแปลงกลับ
plt.subplot(1, 2, 2)
plt.imshow(result[0].reshape(28, 28), cmap=plt.cm.gray, interpolation="nearest")
plt.xlabel(f"{pca.n_components_} Features")  # แสดงจำนวนคุณลักษณะที่ถูกใช้หลังการลดมิติ
plt.title("PCA image")  # ตั้งชื่อภาพ
plt.show()
