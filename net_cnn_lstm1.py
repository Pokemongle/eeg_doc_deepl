import torch.nn as nn
import torch
import numpy as np
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# cnn时间卷积+空间卷积+lstm
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
        self.convspa = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 2), stride=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1)),
            nn.BatchNorm2d(64),  # BatchNormalization层
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 1), stride=(1, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax(dim=1)
        self.gradients = None
        self.feature_maps = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        # print(x.size())
        x = torch.reshape(x, (800, 16, 10, 11))
        # print(x.size())

        # --------- for Grad-CAM ----------
        # self.feature_maps = x
        # # self.feature_maps.requires_grad = True  # 确保需要计算梯度
        # x.register_hook(self.save_gradient)
        # ---------------------------------
        x = self.convspa(x)

        x = torch.reshape(x, (1, 64, 1, 800))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.size())
        # LSTM part
        x = torch.reshape(x, (1, 100, 64))
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


def grad_cam(model, input_data, target_class):
    # 前向传播，得到输出和特征图
    output = model(input_data)
    target_score = output[0, target_class]

    # 清零梯度，并进行反向传播，获取梯度
    model.zero_grad()
    target_score.backward()

    # 获取模型中的梯度和特征图
    gradients = model.gradients  # 梯度，形状: [B, C, H, W]
    feature_maps = model.feature_maps  # 特征图，形状: [B, C, H, W]

    # 对梯度进行全局平均池化
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

    # 按通道重要性对特征图加权
    for i in range(feature_maps.size(1)):
        feature_maps[:, i, :, :] *= pooled_gradients[i]

    # 计算Grad-CAM，按通道求平均值
    heatmap = torch.mean(feature_maps, dim=1).squeeze()
    heatmap = torch.mean(heatmap, dim=0).squeeze()

    # 使用ReLU激活，去除负值
    heatmap = F.relu(heatmap)

    # 归一化热力图
    heatmap /= torch.max(heatmap)

    # 调整热力图大小为10x11 EEG电极分布
    heatmap_resized = heatmap.cpu().detach().numpy()
    print(heatmap_resized.shape)  # 打印检查
    # heatmap_resized = np.mean(heatmap_resized.reshape(800, 10, 11), axis=0)

    return heatmap_resized


def main():
    mode = 1
    if mode == 0:
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
        y = torch.randint(0, 2, (1,))  # 二分类任务，标签为0或1
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
    else:
        # 定义模型结构和超参数
        input_size = 64  # 输入特征维度
        hidden_size = 64  # 隐藏单元数量
        num_layers = 2  # LSTM层数
        output_size = 2  # 输出类别数量
        device = torch.device("cuda:0")
        input_data = torch.randn(110, 1, 1, 2400).to(device)
        model = MyNetwork(input_size, hidden_size, num_layers, output_size).to(device)
        model.train()
        target_class = 1  # 需要可视化的类别

        heatmap = grad_cam(model, input_data, target_class)

        # 显示热力图
        plt.imshow(heatmap, cmap='jet', alpha=0.6)
        plt.colorbar()
        plt.title('Grad-CAM EEG 脑电地形图')
        plt.show()


if __name__ == '__main__':
    main()
    # dl = DataLoader()
