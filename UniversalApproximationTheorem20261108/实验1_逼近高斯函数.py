#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :实验1_拟合高斯函数.py
# @Time :2026/11/8 11:13
# @Author :Li-Dong(LeslieLee)
import torch
import torch.nn as nn
import torch.optim as optim
from 通用近似定理20261108.network import SingleHiddenLayerNet
import numpy as np
import matplotlib.pyplot as plt


def train_network(model, x_train, y_train, epochs=2000, lr=0.01, print_every=200):
    """
    通用的训练循环函数
    """
    # 定义损失函数与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)

    # 训练网络
    loss_history = []
    for epoch in range(epochs):
        # 1. 梯度清零 (PyTorch 默认会累加梯度)
        optimizer.zero_grad()
        # 2. 前向传播计算预测值
        y_pred = model(x_train)
        # 3. 计算损失
        loss = criterion(y_pred, y_train)
        # 4. 反向传播计算梯度 (自动调用你之前手推的那些链式法则)
        loss.backward()
        # 5. 更新参数
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}")

    return loss_history

if __name__ == "__main__":
    # 设置随机种子，保证每次运行结果一致
    torch.manual_seed(42)
    np.random.seed(42)

    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的设备: {device}")

    # 训练集
    a, b = -0.5, 2.5
    N = 100
    x_train = torch.linspace(a, b, N).unsqueeze(1).to(device) # 形状 (100, 1)
    y_train = torch.exp(-(x_train-1)**2 / 2)                  # 形状 (100, 1)
    # 测试集
    x_test = torch.linspace(a, b, 5*N).unsqueeze(1).to(device)
    y_test = torch.exp(-(x_test-1)**2 / 2)

    # 实例化网络
    M = 100  # 隐藏层神经元个数
    model = SingleHiddenLayerNet(input_dim=1, hidden_dim=M, output_dim=1).to(device)

    # ----------训练网络
    loss_history = train_network(model, x_train, y_train, epochs=2000, lr=0.01, print_every=200)

    # ----------评估
    model.eval()
    with torch.no_grad():  # 测试时不需要计算梯度
        y_pred_test = model(x_test)
    # 把数据转回CPU并转成numpy
    x_plot = x_test.cpu().numpy()
    y_true_plot = y_test.cpu().numpy()
    y_pred_plot = y_pred_test.cpu().numpy()
    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(x_plot, y_true_plot, label="True", color="black", linewidth=2)
    plt.plot(x_plot, y_pred_plot, label="Neural Network Prediction", color="red", linestyle="--", linewidth=2)
    plt.title("Single Hidden Layer Network Approximating Gaussian")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

