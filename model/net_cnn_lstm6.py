import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
from torch.utils.data import DataLoader


# cnn时间卷积，用来对比的链式输入
class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyNetwork, self).__init__()
        # CP1
        self.CP1 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.CP2 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(16),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.CP3 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 4), stride=(1, 1)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.CP4 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.CP5 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.CP6 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 4), stride=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.CP7 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.CP8 = nn.Sequential(
            # 59×2400×1
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 4), stride=(1, 1)),
            nn.BatchNorm2d(256),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        )
        self.fc1 = nn.Linear(59 * 256 * 1 * 6, 1000)
        self.fc2 = nn.Linear(1000, 50)
        self.fc3 = nn.Linear(50, 2)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播过程
        x = self.CP1(x)
        # print(x.size())
        x = self.CP2(x)
        # print(x.size())
        x = self.CP3(x)
        # print(x.size())
        x = self.CP4(x)
        # print(x.size())
        x = self.CP5(x)
        # print(x.size())
        x = self.CP6(x)
        # print(x.size())
        x = self.CP7(x)
        # print(x.size())
        x = self.CP8(x)
        # print(x.size())
        x = torch.reshape(x, (x.size(2), -1))
        x = self.fc1(x)
        # print(x.size())
        x = self.fc2(x)
        x = self.fc3(x)
        # x = self.softmax(x)
        # print(x.size())
        return x


def main():
    # 原始输入矩阵
    input_matrix = torch.randn(59, 1, 1, 1, 2400)
    # 重塑为 (110, 1, 1, 2400)
    reshaped_input = torch.reshape(input_matrix, (59, 1, 1, 2400))

    # # 原始输入矩阵
    # input_matrix = torch.randn(30, 1, 1, 1, 4000)
    # # 重塑为 (110, 1, 1, 2400)
    # reshaped_input = torch.reshape(input_matrix, (30, 1, 1, 4000))
    # 定义LSTM超参数
    input_size = 64  # 输入特征维度
    hidden_size = 64  # 隐藏单元数量
    num_layers = 2  # LSTM层数
    output_size = 2  # 输出类别数量

    # # 构建一个随机输入x和对应标签y
    # x = torch.randn(2, 32, 10)  # [batch_size, sequence_length, input_size]
    # print(x.size())
    y = torch.randint(0, 2, (1,))  # 二分类任务，标签为0或1
    # print(f"y_size:{y.size()}")
    # print(y)
    # 创建LSTM模型，并将输入x传入模型计算预测输出
    net = MyNetwork(input_size, hidden_size, num_layers, output_size)
    pred = net(reshaped_input)  # [batch_size, output_size]

    # 定义损失函数和优化器，并进行模型训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    num_epochs = 100

    for epoch in range(num_epochs):
        # 前向传播计算损失函数值
        pred = net(reshaped_input)  # 在每个epoch中重新计算预测输出
        loss = criterion(pred, y.long())

        # 反向传播更新模型参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 输出每个epoch的训练损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


if __name__ == '__main__':
    main()
    # dl = DataLoader()
