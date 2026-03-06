# coding: utf-8
"""
MNIST 数字识别模型训练脚本（PyTorch版本，支持GPU加速）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'origin_data'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class MLP(nn.Module):
    """多层感知机"""
    def __init__(self, input_size=784, hidden_sizes=[100, 100], output_size=10):
        super(MLP, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def train():
    """训练模型"""
    # 参数设置
    max_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    
    # 加载数据集
    print("\n加载MNIST数据集...")
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    
    # 转换为PyTorch张量
    x_train = torch.FloatTensor(x_train)
    t_train = torch.LongTensor(t_train)
    x_test = torch.FloatTensor(x_test)
    t_test = torch.LongTensor(t_test)
    
    # 创建数据加载器
    train_dataset = TensorDataset(x_train, t_train)
    test_dataset = TensorDataset(x_test, t_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练数据: {x_train.shape}")
    print(f"测试数据: {x_test.shape}")
    
    # 创建模型
    model = MLP(input_size=784, hidden_sizes=[100, 100], output_size=10)
    model = model.to(device)
    
    print(f"\n模型结构:")
    print(model)
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练历史
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    # 开始训练
    print(f"\n开始训练 (使用 {device})...")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Test Loss':<12} {'Test Acc':<12}")
    print("-" * 54)
    
    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.4f} {test_loss:<12.4f} {test_acc:<12.4f}")
    
    # 绘制结果
    print("\n绘制训练结果...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失函数曲线
    x = np.arange(len(train_losses))
    axes[0].plot(x, train_losses, 'o-', label='train', markersize=3)
    axes[0].plot(x, test_losses, 's-', label='test', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # 准确率曲线
    axes[1].plot(x, train_accs, 'o-', label='train', markersize=3)
    axes[1].plot(x, test_accs, 's-', label='test', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim([0.95, 1.0])
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results_pytorch.png', dpi=100)
    print("✓ 结果已保存到 training_results_pytorch.png")
    
    # 保存模型
    torch.save(model.state_dict(), 'mnist_model_pytorch.pth')
    print("✓ 模型已保存到 mnist_model_pytorch.pth")
    
    print(f"\n最终测试准确率: {test_accs[-1]:.4f}")
    
    return model

if __name__ == '__main__':
    model = train()
