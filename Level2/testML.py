from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# โหลดข้อมูล Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# แบ่งข้อมูลเป็นชุดการฝึกและชุดการทดสอบ
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างโมเดล
model = LogisticRegression(max_iter=200)

# ฝึกโมเดลด้วยชุดข้อมูลการฝึก
model.fit(X_train, y_train)

# ประเมินโมเดลด้วยชุดข้อมูลการทดสอบ
score = model.score(X_test, y_test)

print(f'Accuracy: {score}')
