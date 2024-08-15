import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# จัดกลุ่ม แปลงให้เป็นตัวเลข
def  cleandata(dataset):
    for column in dataset.columns:
        if dataset[column].dtype == type(object):
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    return dataset

def split_feature_class(dataset,feature):
    features = dataset.drop(feature,axis=1) # เอาข้อมูลทั้งหมดยกเว้น income
    labels = dataset[feature].copy() # เอาเฉพาะข้อมูล income
    return features,labels

dataset = pd.read_csv("adult.csv")
dataset = cleandata(dataset)
print(dataset.head())

# split train,test
training_set,test_set = train_test_split(dataset,test_size=0.2)

# train
train_features,train_label = split_feature_class(training_set,"income")

# test
test_features,test_label = split_feature_class(test_set,"income")

# model
model = GaussianNB()
model.fit(train_features,train_label)

# predict
clf_pred = model.predict(test_features)

print("Accuracy = ",accuracy_score(test_label,clf_pred))

