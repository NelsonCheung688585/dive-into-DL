import matplotlib.pyplot as plt
import torch
import random

# 超参数

# 训练数据的特征维度
num_features = 2
# 训练样本数
num_examples = 1000
# 真实权重
true_w = torch.tensor([2, -3.4])
# 真实偏置
true_b = 4.2
# batch size
batch_size = 10
# 训练次数
num_epochs = 3
# 学习率
learning_rate = 0.03

# 生成数据集
x_data = torch.randn(num_examples, num_features)
y_data = x_data.matmul(true_w.T) + true_b + torch.randn(num_examples) * 0.01

def batch_generator(batch_size, x_data, y_data):
    num_examples = len(x_data)
    indices = list(range(len(x_data)))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch = indices[i:min(i+batch_size, num_examples)]
        yield x_data[batch], y_data[batch]

w = torch.randn(num_features, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for epoch in range(num_epochs):
    for X, y in batch_generator(batch_size, x_data, y_data):
        loss = ((X.matmul(w) + b - y) ** 2 / 2)
        loss.backward(gradient=torch.ones(batch_size))
        # 更新的是data
        w.data = w.data - learning_rate * w.grad / batch_size
        b.data = b.data - learning_rate * b.grad / batch_size
        # 梯度清0
        w.grad.data.zero_()
        b.grad.data.zero_()
    loss = ((x_data.matmul(w) + b - y_data) ** 2 / 2)
    print("epoch %d, loss %f" % (epoch + 1, loss.mean()))