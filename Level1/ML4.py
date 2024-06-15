from scipy.io import  loadmat
import matplotlib.pyplot as plt
mnist_raw = loadmat("mnist-original.mat")


# print(mnist_raw.keys())

# print(mnist_raw)

mnist = {
    "data": mnist_raw["data"].T,
    "target":mnist_raw["label"][0]
}

# print(mnist["data"])
x = mnist["data"]
y = mnist["target"]

number = x[5200]
number_image = number.reshape(28,28)

print(y[5200])
plt.imshow(number_image,cmap=plt.cm.binary,interpolation="nearest")
plt.show()
print(x.shape)
