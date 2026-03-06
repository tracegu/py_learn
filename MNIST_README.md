# MNIST 数字识别模型训练工程

## 项目概述

这个工程使用两层神经网络训练MNIST手写数字识别模型，可以识别0-9的手写数字。

## 项目结构

```
mnist_train.py       - 模型训练脚本
mnist_predict.py     - 模型推理脚本
training_results.png - 训练结果可视化（训练后生成）
mnist_model.pkl      - 训练好的模型参数（训练后生成）
```

## 数据集

使用的是标准MNIST数据集，位置：`D:\python-learn\origin_data\dataset`

- 训练集：60,000张手写数字图像
- 测试集：10,000张手写数字图像
- 图像大小：28×28像素
- 类别：10（数字0-9）

## 快速开始

### 1. 安装依赖

确保已安装以下包：
```bash
pip install numpy matplotlib
```

### 2. 训练模型

在项目目录运行：
```bash
python mnist_train.py
```

训练参数配置：
- 最大轮数（epochs）：20
- 批大小（batch_size）：128
- 学习率（learning_rate）：0.001
- 优化器：Adam
- 网络结构：784 -> 100 -> 100 -> 10

训练完成后会生成：
- `training_results.png` - 包含损失函数和准确率曲线
- `mnist_model.pkl` - 训练好的模型参数

### 3. 模型推理

使用训练好的模型进行推理：
```bash
python mnist_predict.py
```

该脚本会：
- 加载训练好的模型
- 在测试集上评估模型性能
- 显示每个数字的识别准确率
- 生成预测可视化（`predictions_visualization.png`）

## 模型架构

```
输入层（784）
    ↓
隐藏层1（100）- ReLU激活
    ↓
隐藏层2（100）- ReLU激活
    ↓
输出层（10）- Softmax
    ↓
10个类别的概率分布
```

## 使用的工具函数

项目使用了 `origin_data/common` 目录下的以下模块：

- `multi_layer_net.py` - 多层神经网络实现
- `trainer.py` - 训练器，处理模型的训练流程
- `optimizer.py` - 优化器实现（Adam）
- `layers.py` - 网络层实现
- `functions.py` - 激活函数和损失函数

## 超参数调整建议

如需改进模型性能，可以调整以下参数：

1. **网络深度和宽度**：修改 `hidden_size_list=[100, 100]`
   - 例如：`[128, 64]` 或 `[256, 128, 64]`

2. **学习率**：修改 `learning_rate=0.001`
   - 范围：0.0001 到 0.1

3. **批大小**：修改 `batch_size=128`
   - 范围：32 到 256

4. **训练轮数**：修改 `max_epochs=20`
   - 范围：10 到 50

## 预期结果

训练完成后，模型应该能达到：
- 测试集准确率：> 97%
- 训练过程：损失函数逐渐下降，准确率逐渐上升

## 注意事项

1. 首次运行时，如果数据集文件不存在，会自动从网络下载
2. 模型训练时间取决于硬件配置，通常需要几分钟到十几分钟
3. 使用GPU可以大幅加速训练（如果你的环境支持）

## 延伸建议

1. 尝试添加Dropout和BatchNormalization来防止过拟合
2. 使用卷积神经网络（CNN）获得更高的准确率
3. 尝试不同的优化器（SGD, RMSprop等）
4. 实现早停（Early Stopping）机制

## 参考资源

- [MNIST数据集官方网站](http://yann.lecun.com/exdb/mnist/)
- 本项目基于深度学习入门教材的代码框架
