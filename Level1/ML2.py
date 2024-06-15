import pylab
from sklearn import datasets

digit_dataset = datasets.load_digits()

print(digit_dataset.keys())

print(digit_dataset['images'].shape)

print(digit_dataset.target_names)

print(digit_dataset.images[0])
print(digit_dataset.images[0].shape)

print(digit_dataset.images[:5])


for i in range(10):
    print(digit_dataset.target[i])
    pylab.imshow(digit_dataset.images[i],cmap = pylab.cm.gray_r)
    pylab.show()