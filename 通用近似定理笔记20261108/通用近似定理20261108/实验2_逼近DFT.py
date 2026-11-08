#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :实验2_逼近DFT.py
# @Time :2026/11/8 11:48
# @Author :Li-Dong(LeslieLee)
import torch
import torch.nn as nn
import torch.optim as optim
from 通用近似定理20261108.network import SingleHiddenLayerNet
import numpy as np
import matplotlib.pyplot as plt


# 数据生成
def generate_batch(batch_size, x_size, device, mode="fft"):
    # 生成位于[-1, 1]随机实信号
    X = torch.rand(batch_size, x_size, device=device) * 2 - 1
    # fft
    if mode == "fft":
        Y = torch.fft.fft(X, dim=1)  # 形状(batch, N)
    else:  # rfft
        Y = torch.fft.rfft(X, dim=1)  # 形状(batch, floor(N/2)+1)

    Y_expand = torch.cat([Y.real, Y.imag], dim=1)
    return X, Y_expand

# 训练
def train_network(model, x_train, y_train, batch_size=64, epochs=2000, lr=0.01, print_every=200):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_samples = x_train.shape[0]
    loss_history = []

    for epoch in range(epochs):
        # 每个 epoch 打乱数据
        indices = torch.randperm(num_samples)
        x_shuffled = x_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0

        # 分批训练
        for i in range(0, num_samples, batch_size):
            x_batch = x_shuffled[i: i + batch_size]
            y_batch = y_shuffled[i: i + batch_size]

            # 1. 梯度清零
            optimizer.zero_grad()
            # 2. 前向传播
            y_pred = model(x_batch)
            # 3. 计算损失
            loss = criterion(y_pred, y_batch)
            # 4. 反向传播
            loss.backward()
            # 5. 更新参数
            optimizer.step()

            # 累加 loss（乘以当前 batch 的样本数，用于最后算平均）
            epoch_loss += loss.item() * x_batch.size(0)

        # 计算当前 epoch 的平均 loss
        epoch_loss /= num_samples
        loss_history.append(epoch_loss)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.6e}")

    return loss_history


if __name__ == "__main__":
    # 设备与随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"当前使用的设备: {device}")

    # --- 实验配置 ---
    N = 16  # 信号长度
    M = 128  # 隐藏层宽度
    mode = "rfft"  # 你可以改成 "rfft" 测试

    # 动态确定输出维度
    if mode == "fft":
        out_dim = 2 * N
    else:
        K = N // 2 + 1
        out_dim = 2 * K

    print(f"模式: {mode}, 网络输出维度: {out_dim}")

    # --- 预先生成数据集 ---
    print("正在生成数据集...")
    num_train = 10000
    num_test = 500
    X_train, Y_train = generate_batch(num_train, N, device, mode=mode)
    X_test, Y_test = generate_batch(num_test, N, device, mode=mode)

    # --- 实例化模型 ---
    model = SingleHiddenLayerNet(input_dim=N, hidden_dim=M, output_dim=out_dim).to(device)

    # --- 开始训练 ---
    print(f"--- 开始训练 ---")
    losses = train_network(model, X_train, Y_train, batch_size=64, epochs=500, lr=1e-3, print_every=100)

    # --- 画 Loss 曲线 ---
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.yscale('log')
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # --- 测试与可视化 ---
    model.eval()
    with torch.no_grad():
        for idx in [0, 10, 100, 300]:
            # 取一个测试样本
            X_sample = X_test[idx:idx+1, :]
            Y_true = Y_test[idx:idx+1, :]
            Y_pred = model(X_sample)

            # 转回numpy
            pred_real = Y_pred[0, :Y_pred.shape[1] // 2].cpu().numpy()
            pred_imag = Y_pred[0, Y_pred.shape[1] // 2:].cpu().numpy()
            true_real = Y_true[0, :Y_true.shape[1] // 2].cpu().numpy()
            true_imag = Y_true[0, Y_true.shape[1] // 2:].cpu().numpy()
            # 计算幅值
            pred_amp = np.sqrt(pred_real ** 2 + pred_imag ** 2)
            true_amp = np.sqrt(true_real ** 2 + true_imag ** 2)

            # 画幅值谱对比
            plt.figure(figsize=(10, 5))
            k = np.arange(len(true_amp))
            plt.plot(k, true_amp, 'ko-', label='True', linewidth=2)
            plt.plot(k, pred_amp, 'r--x', label='Pred', linewidth=2)
            plt.title("Amplitude Spectrum", fontsize=14)
            plt.xlabel("Frequency index k", fontsize=12)
            plt.ylabel("Amplitude", fontsize=12)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.5)
            plt.tight_layout()
            plt.show()