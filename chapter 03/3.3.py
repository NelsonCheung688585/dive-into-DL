import torch

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
num_epochs = 4
# 学习率
learning_rate = 0.03

# 生成数据集
x_data = torch.randn(num_examples, num_features)
y_data = x_data.matmul(true_w.T) + true_b + torch.randn(num_examples) * 0.01

dataset = torch.utils.data.TensorDataset(x_data, y_data)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

class Net(torch.nn.Module):
    def __init__(self, feature_number):
        super(Net, self).__init__()
        self.Linear = torch.nn.Linear(feature_number, 1, bias=True)

    def forward(self, x):
        x = self.Linear(x)
        return x

net = Net(num_features)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(num_epochs):
    for batch_x, batch_y in data_loader:
        optimizer.zero_grad()
        output = net(batch_x).reshape(batch_size)
        #print(output.shape, batch_y.shape)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
    output = net(x_data).reshape(num_examples)
    loss = criterion(y_data, output)
    print("epoch: %d, loss: %f" % (epoch, loss.item()))