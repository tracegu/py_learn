# coding: utf-8
"""
MNIST 模型推理脚本
加载训练好的模型并进行推理
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'origin_data'))

import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
import matplotlib.pyplot as plt

def predict_single_image(network, image):
    """对单个图像进行预测"""
    image = image.reshape(1, -1)
    output = network.predict(image)
    predicted_label = np.argmax(output, axis=1)
    confidence = np.max(output, axis=1)
    return predicted_label[0], confidence[0]

def visualize_predictions(network, x_test, t_test, num_samples=10):
    """可视化预测结果"""
    plt.figure(figsize=(15, 3))
    
    for i in range(num_samples):
        # 随机选择一个测试样本
        idx = np.random.randint(0, len(x_test))
        image = x_test[idx]
        true_label = t_test[idx]
        
        # 预测
        pred_label, confidence = predict_single_image(network, image)
        
        # 绘制
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'Pred: {pred_label} Conf: {confidence:.2f}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions_visualization.png')
    print("预测可视化已保存到 predictions_visualization.png")
    plt.show()

def evaluate_model(network, x_test, t_test):
    """评估模型性能"""
    predictions = []
    for image in x_test:
        pred_label, _ = predict_single_image(network, image)
        predictions.append(pred_label)
    
    predictions = np.array(predictions)
    accuracy = np.mean(predictions == t_test)
    
    # 计算每个数字的准确率
    print("\n各数字识别准确率:")
    for digit in range(10):
        mask = t_test == digit
        digit_acc = np.mean(predictions[mask] == t_test[mask])
        count = np.sum(mask)
        print(f"数字 {digit}: {digit_acc:.4f} ({int(count)} 样本)")
    
    return accuracy

def main():
    """主函数"""
    # 加载模型
    if not os.path.exists('mnist_model.pkl'):
        print("错误: 找不到 mnist_model.pkl, 请先运行 mnist_train.py 训练模型")
        return
    
    print("加载模型...")
    with open('mnist_model.pkl', 'rb') as f:
        params = pickle.load(f)
    
    # 创建网络并加载参数
    network = MultiLayerNet(
        input_size=784,
        hidden_size_list=[100, 100],
        output_size=10
    )
    network.params = params
    
    # 加载测试数据
    print("加载MNIST数据集...")
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
    
    # 评估模型
    print("\n评估模型性能...")
    test_acc = evaluate_model(network, x_test, t_test)
    print(f"\n总体测试精度: {test_acc:.4f}")
    
    # 可视化预测结果
    print("\n可视化预测结果...")
    visualize_predictions(network, x_test, t_test)

if __name__ == '__main__':
    main()
