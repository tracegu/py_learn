# coding: utf-8
"""
MNIST 数字识别模型训练脚本
使用两层神经网络训练MNIST数据集
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'origin_data'))

import numpy as np
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from common.optimizer import Adam
import matplotlib.pyplot as plt

def train():
    """训练模型"""
    # 设置参数
    max_epochs = 20
    batch_size = 128
    learning_rate = 0.001
    
    # 加载数据集
    print("Loading MNIST dataset...")
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    print(f"训练数据形状: {x_train.shape}")
    print(f"测试数据形状: {x_test.shape}")
    
    # 创建模型
    # 输入层: 784 (28*28)
    # 隐藏层: [100, 100]
    # 输出层: 10 (0-9数字)
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100],
        output_size=10
    )
    
    # 创建优化器
    optimizer = Adam(lr=learning_rate)
    
    # 创建训练器
    trainer = Trainer(
        network=network,
        x_train=x_train,
        t_train=t_train,
        x_test=x_test,
        t_test=t_test,
        epochs=max_epochs,
        mini_batch_size=batch_size,
        optimizer=optimizer,
        evaluate_sample_num_per_epoch=1000
    )
    
    # 开始训练
    print("\nStarting training...")
    trainer.train()
    
    # 获取训练结果
    markers = {'train': 'o', 'test': 's'}
    x = np.arange(len(trainer.train_loss_list))
    
    # 绘制损失函数曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(x, trainer.train_loss_list, marker='o', label='train', markersize=3)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim([0, max(trainer.train_loss_list) * 1.1])
    plt.legend()
    plt.grid(True)
    
    # 绘制精度曲线
    plt.subplot(1, 2, 2)
    plt.plot(x, trainer.train_acc_list, marker='o', label='train', markersize=3)
    plt.plot(x, trainer.test_acc_list, marker='s', label='test', markersize=3)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim([0.95, 1.0])
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    print("\n训练结果已保存到 training_results.png")
    
    # 在测试集上评估
    test_acc = network.accuracy(x_test, t_test)
    print(f"\n最终测试精度: {test_acc:.4f}")
    
    # 保存模型参数
    import pickle
    with open('mnist_model.pkl', 'wb') as f:
        pickle.dump(network.params, f)
    print("模型已保存到 mnist_model.pkl")
    
    return network

if __name__ == '__main__':
    network = train()
