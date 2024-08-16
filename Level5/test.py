import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

# ฟังก์ชันในการแปลงข้อมูลเชิงสัญลักษณ์ให้เป็นตัวเลข
def cleandata(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

# ฟังก์ชันในการแยกข้อมูลคุณลักษณะ (features) และป้ายกำกับ (labels)
def split_feature_class(dataset, feature):
    features = dataset.drop(feature, axis=1)  # ลบคอลัมน์ target ออกจากชุดข้อมูล
    labels = dataset[feature].copy()          # คัดลอกคอลัมน์ target
    return features, labels

# โหลดชุดข้อมูล
dataset = pd.read_csv("adult.csv")
dataset = cleandata(dataset)
# ตรวจสอบชื่อคอลัมน์ทั้งหมดใน dataset
print(dataset.columns)

# เลือกเฉพาะคุณลักษณะที่ต้องการ (ตัวอย่างเช่นใช้เฉพาะ features ที่เข้าใจว่ามีผลต่อการทำนาย)
selected_features = ['age', 'education', 'hours-per-week', 'capital-gain', 'capital-loss']  # ปรับชื่อให้ตรงกับที่มีอยู่จริง

dataset_selected = dataset[selected_features + ['income']]  # เลือกคุณลักษณะ + เป้าหมาย

# แบ่งข้อมูลออกเป็นชุดฝึกสอนและชุดทดสอบ
training_set, test_set = train_test_split(dataset_selected, test_size=0.2, random_state=42)

# แยกข้อมูลออกเป็น features และ labels
train_features, train_label = split_feature_class(training_set, "income")
test_features, test_label = split_feature_class(test_set, "income")

# สร้างและฝึกโมเดล
model = GaussianNB()
model.fit(train_features, train_label)

# ทำการพยากรณ์และประเมินผล
clf_pred = model.predict(test_features)
print("Accuracy = ", accuracy_score(test_label, clf_pred))
