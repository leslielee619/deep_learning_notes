#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :实验3_逼近传热方程.py
# @Time :2026/11/8 15:32
# @Author :Li-Dong(LeslieLee)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# ================= 1. 单隐藏层网络结构 =================
class PINN(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # 严格的单隐藏层网络：输入层(2) -> 隐藏层(hidden_dim) -> 输出层(1)
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),  # 只有一个非线性激活函数
            nn.Linear(hidden_dim, 1)  # 输出层（线性）
        )

    def forward(self, x, t):
        # 将 x 和 t 拼接成 (N, 2) 的输入
        inputs = torch.cat([x, t], dim=1)
        return self.net(inputs)


# ================= 2. 数据生成 =================
def generate_data(N_f=2000, N_bc=200, L=1.0, T=0.3, device="cpu"):
    # 内部配点 (x, t)，需要计算梯度，所以 requires_grad=True
    x_f = (torch.rand(N_f, 1, device=device) * L).requires_grad_(True)
    t_f = (torch.rand(N_f, 1, device=device) * T).requires_grad_(True)

    # 初始条件点 (x, t=0)
    x_0 = torch.rand(N_bc, 1, device=device) * L
    t_0 = torch.zeros(N_bc, 1, device=device)

    # 边界条件点 (x=0 和 x=L)
    t_b = torch.rand(N_bc, 1, device=device) * T
    x_b0 = torch.zeros(N_bc, 1, device=device)
    x_b1 = torch.full((N_bc, 1), L, device=device)

    return x_f, t_f, x_0, t_0, x_b0, x_b1, t_b


# ================= 3. 损失计算 =================
def compute_loss(model, x_f, t_f, x_0, t_0, x_b0, x_b1, t_b):
    # 1. PDE 残差 (利用 Autograd 求导)
    u = model(x_f, t_f)

    # 一阶导 u_t, u_x
    u_t = torch.autograd.grad(u, t_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x_f, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    # 二阶导 u_xx (对 u_x 再次求导)
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # 热传导方程: u_t = alpha * u_xx (这里 alpha = 1)
    loss_pde = torch.mean((u_t - u_xx) ** 2)

    # 2. 初始条件: u(x,0) = sin(pi * x)
    u_0 = model(x_0, t_0)
    target_0 = torch.sin(np.pi * x_0)
    loss_init = torch.mean((u_0 - target_0) ** 2)

    # 3. 边界条件 (齐次 Dirichlet): u(0,t) = 0, u(L,t) = 0
    u_b0 = model(x_b0, t_b)
    u_b1 = model(x_b1, t_b)
    loss_bc = torch.mean(u_b0 ** 2) + torch.mean(u_b1 ** 2)

    return loss_pde, loss_init, loss_bc


# ================= 4. 训练函数 =================
def train(model, data, epochs=10000, lr=1e-3, lambda_init=10.0, lambda_bc=10.0):
    # 解包数据
    x_f, t_f, x_0, t_0, x_b0, x_b1, t_b = data

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_history = []

    print("开始 PINN 训练...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        loss_pde, loss_init, loss_bc = compute_loss(model, x_f, t_f, x_0, t_0, x_b0, x_b1, t_b)
        loss = loss_pde + lambda_init * loss_init + lambda_bc * loss_bc

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | Total: {loss.item():.4e} | PDE: {loss_pde.item():.4e} | Init: {loss_init.item():.4e} | BC: {loss_bc.item():.4e}")

    return loss_history


# ================= 5. 测试与可视化 =================
def evaluate_and_plot(model, L=1.0, T=0.3, device="cpu"):
    # 解析解: u(x,t) = exp(-pi^2 * t) * sin(pi * x)
    def exact_solution(x, t):
        return np.exp(-np.pi ** 2 * t) * np.sin(np.pi * x)

    model.eval()
    with torch.no_grad():
        x_test = np.linspace(0, L, 100)
        t_test = np.linspace(0, T, 100)
        X_grid, T_grid = np.meshgrid(x_test, t_test)

        # 展平并转为 Tensor
        x_flat = torch.tensor(X_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)
        t_flat = torch.tensor(T_grid.flatten(), dtype=torch.float32, device=device).unsqueeze(1)

        # 预测并 reshape 回网格形状
        U_pred = model(x_flat, t_flat).cpu().numpy().reshape(X.shape)
        U_exact = exact_solution(X_grid, T_grid)

        error = np.abs(U_pred - U_exact)
        print(f"\n最大绝对误差 (Max Error): {np.max(error):.6e}")

        # 画图
        fig = plt.figure(figsize=(15, 5))

        ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        ax1.plot_surface(X_grid, T_grid, U_exact, cmap='viridis')
        ax1.set_title("Exact Solution")
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')

        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax2.plot_surface(X_grid, T_grid, U_pred, cmap='plasma')
        ax2.set_title("PINN Prediction")
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')

        ax3 = fig.add_subplot(1, 3, 3)
        c = ax3.contourf(X_grid, T_grid, error, levels=20, cmap='Reds')
        fig.colorbar(c, ax=ax3)
        ax3.set_title("Absolute Error")
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')

        plt.tight_layout()
        plt.show()


# ================= 6. 主程序运行 =================
if __name__ == "__main__":
    # 设置设备与随机种子
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    np.random.seed(42)
    print(f"当前使用的设备: {device}")

        # 物理参数
    L = 1.0
    T_max = 0.3

    # 1. 生成数据
    data = generate_data(N_f=3000, N_bc=300, L=L, T=T_max, device=device)

    # 2. 初始化单隐藏层模型 (如果效果不好，可以把 hidden_dim 调到 128)
    model = PINN(hidden_dim=64).to(device)

    # 3. 训练
    losses = train(model, data, epochs=10000, lr=1e-3)

    # Loss曲线
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.yscale('log')
    plt.title("Training Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # 4. 评估与画图
    evaluate_and_plot(model, L=L, T=T_max, device=device)