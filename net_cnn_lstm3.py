import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
from torch.utils.data import DataLoader


# 毕设模型
# 定义一个LSTM模型
class MyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MyNetwork, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            # 1X800X64, 10X11
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 9), stride=(1, 3), padding=(0, 4)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            # 1X800X32, 10X11
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 1)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            # 1X800X16, 10X11
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1)),
            nn.BatchNorm2d(16),  # BatchNormalization层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 13), stride=(1, 2), padding=(0, 6)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.conv2_3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2)),
            nn.BatchNorm2d(32),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.convspa = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 2), stride=(1, 1)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True)  # ReLU激活函数
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            # nn.ReLU(inplace=True)  # ReLU激活函数
        )
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
        #     nn.BatchNorm2d(128),  # BatchNormalization层
        #     nn.ReLU(inplace=True),  # ReLU激活函数
        #     nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
        #     nn.BatchNorm2d(64),  # BatchNormalization层
        #     nn.ReLU(inplace=True)  # ReLU激活函数
        # )
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
        #     nn.BatchNorm2d(128),  # BatchNormalization层
        #     nn.ReLU(inplace=True),  # ReLU激活函数
        #     nn.MaxPool2d(kernel_size=(1, 2)),  # 添加 1x2 的 MaxPooling 层
        #     nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1)),
        #     nn.BatchNorm2d(64),  # BatchNormalization层
        #     nn.ReLU(inplace=True)  # ReLU激活函数
        # )
        # self.fc1 = nn.Linear(64 * 25, 30)
        # self.fc2 = nn.Linear(30, 3)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x3 = self.conv2_3(x)
        x = torch.cat((x1, x2, x3), dim=1)
        x = torch.reshape(x, (400, 96, 10, 11))
        x = self.convspa(x)
        # print(x.size())
        x = torch.reshape(x, (1, 64, 1, 400))
        x = self.conv3(x)
        # print(x.size())
        x = self.conv4(x)
        # Original CNN part
        # x = self.conv5(x)
        # x = self.conv6(x)
        # x = torch.reshape(x, (x.size(0), -1))  # Flatten the output of the last convolutional layer
        # self.fc1 = nn.Linear(64 * 25, 30)
        # self.fc2 = nn.Linear(30, 3)
        # LSTM part
        x = torch.reshape(x, (1, 100, 64))
        # 初始化隐藏状态h0, c0为全0向量
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # # 初始化隐藏状态h0, c0为He正态分布向量
        h0 = init.kaiming_normal_(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        c0 = init.kaiming_normal_(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(x.device)
        # 将输入x和隐藏状态(h0, c0)传入LSTM网络
        out, _ = self.lstm(x, (h0, c0))
        # print(out[:,-1,:].size())
        # 取最后一个时间步的输出作为LSTM网络的输出
        out = self.fc(out[:, -1, :])
        # print(out.size())
        # print(out)
        # out = self.softmax(out)
        return out


def main():
    # 原始输入矩阵
    input_matrix = torch.randn(10, 11, 1, 1, 2400)

    # 重塑为 (110, 1, 1, 2400)
    reshaped_input = torch.reshape(input_matrix, (110, 1, 1, 2400))

    # 定义LSTM超参数
    input_size = 64  # 输入特征维度
    hidden_size = 64  # 隐藏单元数量
    num_layers = 2  # LSTM层数
    output_size = 2  # 输出类别数量

    # # 构建一个随机输入x和对应标签y
    # x = torch.randn(2, 32, 10)  # [batch_size, sequence_length, input_size]
    # print(x.size())
    y = torch.randint(0, 3, (1,))  # 二分类任务，标签为0或1
    # print(f"y_size:{y.size()}")
    # print(y)
    # 创建LSTM模型，并将输入x传入模型计算预测输出
    net = MyNetwork(input_size, hidden_size, num_layers, output_size)
    pred = net(reshaped_input)  # [batch_size, output_size]

    # 定义损失函数和优化器，并进行模型训练
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
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