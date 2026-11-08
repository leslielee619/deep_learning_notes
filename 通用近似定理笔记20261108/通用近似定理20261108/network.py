#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName :network.py
# @Time :2026/11/8 14:46
# @Author :Li-Dong(LeslieLee)
import torch.nn as nn

class SingleHiddenLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=nn.Tanh):
        super().__init__()
        self.activation = activation()  # 激活函数
        self.hidden = nn.Linear(input_dim, hidden_dim) # 第一层：W 和 b (输入层到隐藏层)
        self.output = nn.Linear(hidden_dim, output_dim) # 第二层：v 和 c (隐藏层到输出层)

    def forward(self, x):
        z = self.hidden(x)        # z = Wx + b
        a = self.activation(z)    # a = phi(z)
        out = self.output(a)      # out = v^T a + c
        return out
