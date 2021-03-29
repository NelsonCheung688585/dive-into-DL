import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, in_features):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(in_features, 1, bias=True)

    def forward(self, x):
        x = self.Linear1(x)
        return x

in_features = 2
input = torch.randn(8, in_features, requires_grad=True)
net = Net(in_features)
output = net(input)
print(output)