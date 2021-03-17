import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# 获取训练集
mnist_train = torchvision.datasets.FashionMNIST(root='../datasets/FashionMNIST', train=True, download=False,
                                                transform=transforms.ToTensor())
# 获取测试集
mnist_test = torchvision.datasets.FashionMNIST(root='../datasets/FashionMNIST', train=False, download=False,
                                               transform=transforms.ToTensor())

print("amount of train data: %d" % len(mnist_train))
print("amount of test data: %d" % len(mnist_test))

feature, label = mnist_train[0]
print(feature.shape, feature.dtype)
print(label)


fig = plt.figure()
img = torchvision.utils.make_grid(feature).numpy()
fig.add_subplot(1, 2, 1)
plt.imshow(np.transpose(img, (1, 2, 0)))

feature, label = mnist_train[1]
img = torchvision.utils.make_grid(feature).numpy()
fig.add_subplot(1, 2, 2)
plt.imshow(np.transpose(img, (1, 2, 0)))

plt.show()
