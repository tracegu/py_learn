import os
import sys
from loguru import logger
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import origin_data.dataset.mnist as mnist
import pickle
# x = np.arange(0, 10, 0.1)
# y = np.sin(x)
# y2 = np.cos(x)
# plt.plot(x, y)
# plt.plot(x, y2)
# plt.legend(['sin(x)', 'cos(x)'])d
# plt.show()

# picture_file = "D:\\python-learn\\origin_data\\dataset\\lena.png"
# img = imread(picture_file)
# plt.imshow(img)
# plt.show()
def step_function(x):
    y = x > 0
    return y.astype(np.int64)

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def relu(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
def luoji(x1, x2):
    X = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    result = np.sum(X * w) + b
    if result <= 0:
        return 0
    else:
        return 1

def get_data():
    (x_train, t_train), (x_test, t_test) = mnist.load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("D:\\python-learn\\origin_data\\ch03\\sample_weight.pkl", 'rb') as f:
        network =  pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_gradient(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)

def func_1(x):
    return 0.01*x**2 + x**2

def func_2(x):
    return x[0]**2 + x[1]**2

if __name__ == "__main__":
    # x = np.array([-5.0, 5.0, 0.1])
    # x = np.arange(-10.0, 10.0, 0.1)
    # y1 = sigmoid(x)
    # y2 = step_function(x)
    # y3 = relu(x)
    # y4 = -relu(x-5)

    # plt.plot(x, y1)
    # plt.plot(x, y2, '--')
    # plt.plot(x, y3)
    # plt.plot(x, y4, '--')
    # plt.plot(x, y3+y4)
    # plt.legend(['relu'])
    # plt.ylim(-10, 10)
    # plt.show()
    # A = np.array([[1, 2, 3], [4, 5, 6]])
    # B = np.array([[1, 2], [3, 4], [5, 6]])
    # print(A.shape)
    # print(B.shape)
    # print(np.dot(A, B))
    
    # x = np.array([1.0, 2.0])
    # w = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    # b = np.array([0.1, 0.2, 0.3])
    # y1 = np.dot(x, w) + b
    # print(y1)   #第一层神经元线性

    # w2 = np.array([[3.0, 0.4], [4.0, -0.5], [5.0, 0.6]])
    # b2 = np.array([0.1, 0.2])
    # print(np.dot(y1, w2) + b2)
    # y2 = sigmoid(np.dot(y1, w2) + b2)
    # print(y2)   #第二层神经元线性+relum

    # x, t = get_data()
    # network = init_network()
    # batch_size = 100

    # accuracy_cnt = 0
    # for i in range(0, len(x), batch_size):
    #     x_batch = x[i:i+batch_size]
    #     y_batch = predict(network, x_batch)
    #     p_batch = np.argmax(y_batch, axis=1)
    #     accuracy_cnt += np.sum(p_batch == t[i:i+batch_size])
    # logger.info("Accuracy:" + str(float(accuracy_cnt) / len(x)))

    # 生成连续的点进行测试
    x0 = np.arange(-5, 5, 0.1)
    x1 = np.arange(-5, 5, 0.1)
    X, Y = np.meshgrid(x0, x1)
    
    # 计算函数值
    Z = X**2 + Y**2
    
    # 绘制等高线图
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 2, 1)
    contour = plt.contour(X, Y, Z)
    plt.clabel(contour, inline=True, fontsize=8)
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title('func_2(x) = x0^2 + x1^2 等高线图')
    plt.colorbar(contour)
    
    # 绘制3D曲面图
    ax = plt.subplot(1, 2, 2, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('func_2(x)')
    ax.set_title('func_2(x) 三维曲面')
    
    plt.tight_layout()
    plt.show()
    
    # 测试几个特定点的梯度
    # print("\n测试特定点的梯度：")
    # test_points = [
    #     np.array([3.0, 4.0]),
    #     np.array([1.0, 1.0]),
    #     np.array([0.0, 0.0]),
    #     np.array([2.0, 3.0])
    # ]
    
    # for point in test_points:
    #     result = func_2(point)
    #     gradient = numerical_gradient(func_2, point)
    #     print(f"func_2({point}) = {result}, gradient = {gradient}")
    