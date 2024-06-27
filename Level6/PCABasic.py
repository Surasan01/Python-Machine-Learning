from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sd
x,y = make_blobs(n_samples=100,n_features=10)

print("Before = ",x.shape)
# print(y)

pca = PCA(n_components=4)

# pca.fit(x)
# x = pca.transform(x)
# รวม
x=pca.fit_transform(x)
print("After = ",x.shape)
# print(pca.n_components_)
# print(pca.explained_variance_ratio_)

df = pd.DataFrame({'var':pca.explained_variance_ratio_,'pc':['PC1','PC2','PC3','PC4']})
sd.barplot(x='pc',y='var',data=df,color='r')
plt.show()

