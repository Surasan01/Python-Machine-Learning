from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# สร้างข้อมูลจำลองที่มีศูนย์กลาง (centers) 4 กลุ่ม จำนวนข้อมูลทั้งหมด 300 ตัวอย่าง
x, y = make_blobs(n_samples=300, centers=4, cluster_std=0.5, random_state=0)

# สร้างข้อมูลชุดใหม่เพื่อใช้ในการทดสอบการจัดกลุ่ม จำนวน 10 ตัวอย่าง
x_test, y_test = make_blobs(n_samples=10, centers=4, cluster_std=0.5, random_state=0)

# แสดงข้อมูล
print(x)  # ข้อมูลคุณลักษณะทั้งหมด
print(x[:, 0])  # ข้อมูลคุณลักษณะแกน X
print(x[:, 1])  # ข้อมูลคุณลักษณะแกน Y
print(y)  # ป้ายกำกับ (labels) ที่บอกว่าข้อมูลอยู่ในกลุ่มไหน (ใช้สำหรับทดสอบเท่านั้น)

# สร้างโมเดล KMeans สำหรับการจัดกลุ่ม (clustering) โดยกำหนดให้แบ่งเป็น 4 กลุ่ม
model = KMeans(n_clusters=4)

# ฝึกโมเดลด้วยข้อมูลที่สร้างขึ้นมา
model.fit(x)

# ทำนายกลุ่มของข้อมูลที่ใช้ในการฝึก (x)
y_pred = model.predict(x)

# ทำนายกลุ่มของข้อมูลใหม่ที่ใช้ในการทดสอบ (x_test)
y_pred_new = model.predict(x_test)

# ดึงตำแหน่งของจุดศูนย์กลางของแต่ละกลุ่ม (centroids) ที่คำนวณได้จากโมเดล
centers = model.cluster_centers_

# แสดงการกระจายตัวของข้อมูลตามกลุ่มที่ทำนายได้ (y_pred)
plt.scatter(x[:, 0], x[:, 1], c=y_pred)

# แสดงการกระจายตัวของข้อมูลทดสอบ (x_test) พร้อมแสดงกลุ่มที่ทำนายได้ (y_pred_new) และขนาดจุดที่ใหญ่กว่า
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_pred_new, s=120)

# แสดงตำแหน่งของจุดศูนย์กลางของแต่ละกลุ่มด้วยสีที่แตกต่างกัน
print(centers)
print(y_pred)

# แสดงจุดศูนย์กลางของแต่ละกลุ่มด้วยสีกำหนดไว้ พร้อมทั้งระบุว่าเป็นศูนย์กลางของกลุ่มไหน
plt.scatter(centers[0, 0], centers[0, 1], c='blue', label='Centroid 1')
plt.scatter(centers[1, 0], centers[1, 1], c='green', label='Centroid 2')
plt.scatter(centers[2, 0], centers[2, 1], c='red', label='Centroid 3')
plt.scatter(centers[3, 0], centers[3, 1], c='black', label='Centroid 4')

# แสดงตำนาน (legend) บนกราฟเพื่อให้รู้ว่าจุดสีไหนเป็นศูนย์กลางของกลุ่มอะไร
plt.legend(frameon=True)

# แสดงกราฟที่สร้างขึ้นมา
plt.show()
