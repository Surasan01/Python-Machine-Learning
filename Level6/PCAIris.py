from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# โหลดชุดข้อมูล Iris
iris = sb.load_dataset('iris')

# แยกข้อมูล features และ target
x = iris.drop('species', axis=1)  # คุณลักษณะ 4 ตัว
y = iris['species']  # ชนิดของดอกไอริส

print("Before = ", x.shape)

# ทำ PCA ลดมิติข้อมูลเหลือ 3 มิติ
pca = PCA(n_components=3)
x_pca = pca.fit_transform(x)
print("After = ", x_pca.shape)

# เพิ่มคอลัมน์ PCA1, PCA2, PCA3 ใน DataFrame x
x['PCA1'] = x_pca[:, 0]
x['PCA2'] = x_pca[:, 1]
x['PCA3'] = x_pca[:, 2]
print(x)

# แบ่งข้อมูลเป็นชุดฝึกสอนและชุดทดสอบ
x_train, x_test, y_train, y_test = train_test_split(x, y)

# เลือกเฉพาะคอลัมน์ PCA1, PCA2, PCA3 สำหรับการฝึกสอนและทดสอบ
x_train = x_train.loc[:, ['PCA1', 'PCA2', 'PCA3']]
x_test = x_test.loc[:, ['PCA1', 'PCA2', 'PCA3']]

# แสดงกราฟแท่งแสดงความแปรปรวนที่อธิบายได้ของแต่ละองค์ประกอบหลัก
df = pd.DataFrame({'var': pca.explained_variance_ratio_, 'pc': ['PC1', 'PC2', 'PC3']})
sb.barplot(x='pc', y='var', data=df, color='r')
plt.show()  # ต้องเพิ่มคำสั่งนี้เพื่อแสดงกราฟ

# สร้างและฝึกโมเดล Gaussian Naive Bayes
model = GaussianNB()
model.fit(x_train, y_train)

# ทำนายผลและคำนวณความแม่นยำ
y_pred = model.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred))
