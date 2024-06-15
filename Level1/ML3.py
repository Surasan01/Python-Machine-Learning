import matplotlib.pyplot as plt
from sklearn import datasets

digit_dataset = datasets.load_digits()

for i in range(10):
    print(digit_dataset.target[i])
    plt.imshow(digit_dataset.images[i],cmap = plt.get_cmap('gray'))
    plt.show()