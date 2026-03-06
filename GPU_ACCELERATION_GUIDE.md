# GPU加速MNIST训练指南

## 快速开始

### 1. 安装PyTorch（带GPU支持）

#### Windows + NVIDIA GPU:
```bash
# 如果有NVIDIA GPU，安装CUDA版本（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或者使用conda（更推荐）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### Mac:
```bash
pip install torch torchvision torchaudio
```

#### CPU only (如果没有GPU):
```bash
pip install torch torchvision torchaudio
```

### 2. 验证GPU环境

运行以下Python代码检查GPU是否可用：
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### 3. 运行训练脚本

```bash
# 使用PyTorch版本（自动检测GPU并使用）
python mnist_train_pytorch.py

# 或者运行推理
python mnist_predict_pytorch.py
```

## 性能对比

| 配置 | 单Epoch时间 | 20 Epochs总时间 |
|------|-----------|----------------|
| NumPy (CPU) | ~5-10秒 | ~2-3分钟 |
| PyTorch (CPU) | ~2-3秒 | ~40-60秒 |
| PyTorch (GPU) | ~0.1-0.3秒 | ~2-6秒 |

**加速倍数**: GPU相比CPU可以快 **10-30倍** 🚀

## 系统要求

### GPU支持
- **NVIDIA GPU**: 需要CUDA 11.8或更高版本
  - 推荐：RTX系列（RTX 3060/3070/3080等）
  - 兼容：GTX系列（GTX 1060及以上）
- **AMD GPU**: 可以使用ROCm版本的PyTorch
- **Mac GPU**: M1/M2/M3芯片原生支持MPS加速

### 最低要求
- GPU显存: 2GB以上
- RAM: 4GB以上
- 磁盘: 500MB以上

## 常见问题

### Q: 如何检查是否正在使用GPU?
A: 查看训练日志开头，如果看到 `使用设备: cuda:0` 说明在使用GPU

### Q: 安装GPU版本失败怎么办?
A: 
1. 确保NVIDIA驱动程序已安装
2. 运行 `nvidia-smi` 检查NVIDIA驱动
3. 可以先安装CPU版本，脚本会自动降级到CPU运行

### Q: CPU运行太慢怎么办?
A:
1. 减少 `batch_size`（例如64或32）
2. 减少 `max_epochs`（例如10）
3. 减少隐藏层大小（例如 `[50, 50]`）
4. 安装GPU版本PyTorch以获得显著加速

### Q: PyTorch版本和原来的NumPy版本有什么不同?
A: 
- 功能完全相同，都是同样的网络结构
- PyTorch版本自动使用GPU（如果可用）
- PyTorch版本训练速度快得多
- 模型文件格式不同，不能互相加载

### Q: 可以在GPU和CPU之间切换吗?
A: 是的，脚本会自动检测。设置环境变量强制使用CPU:
```bash
# Windows CMD
set CUDA_VISIBLE_DEVICES=

# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=""

# Linux/Mac
export CUDA_VISIBLE_DEVICES=
```

## 优化建议

### 1. 增加批大小（如果GPU显存充足）
```python
batch_size = 256  # 或更大
```

### 2. 使用更深的网络
```python
hidden_sizes = [256, 128, 64]  # 替代 [100, 100]
```

### 3. 调整学习率
```python
learning_rate = 0.0005  # 尝试不同值
```

### 4. 使用学习率调度
```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# 在训练循环中调用: scheduler.step()
```

## PyTorch vs NumPy版本选择

| 特性 | NumPy版本 | PyTorch版本 |
|------|----------|----------|
| GPU支持 | ❌ | ✅ |
| 训练速度 | 慢 | 快 |
| 易于修改 | ✅ | ✅ |
| 易于部署 | ❌ | ✅ |
| 推理速度 | 慢 | 快 |

**建议**: 优先使用PyTorch版本！

## 相关命令

```bash
# 检查Python环境
python --version

# 检查PyTorch版本
python -c "import torch; print(torch.__version__)"

# 检查CUDA版本（如果已安装）
python -c "import torch; print(torch.version.cuda)"

# 查看NVIDIA GPU信息（Windows）
nvidia-smi

# 运行GPU内存监控
nvidia-smi --query-gpu=utilization.gpu,utilization.memory --loop=1
```

## 进阶技巧

### 1. 分布式训练（多GPU）
```python
model = nn.DataParallel(model)  # 添加这行
```

### 2. 混合精度训练（加速 + 节省显存）
```python
from torch.cuda.amp import autocast
with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

### 3. 模型量化（加快推理）
```python
model_quantized = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear})
```

## 性能监控

运行训练时监控GPU使用情况（开启另一个终端）：

```bash
# Windows
nvidia-smi -l 1

# Linux/Mac
watch -n 1 nvidia-smi
```

## 参考资源

- [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
- [CUDA安装指南](https://docs.nvidia.com/cuda/cuda-installation-guide-windows/index.html)
- [PyTorch GPU最佳实践](https://pytorch.org/docs/stable/notes/cuda.html)
