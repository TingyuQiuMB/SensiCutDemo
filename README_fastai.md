# SensiCut Fast.ai 版本训练脚本

这是使用 fast.ai 框架重写的 SensiCut 材料识别训练脚本。相比原始的 PyTorch 版本，fast.ai 版本提供了更简洁的 API 和更多内置功能。

## 文件说明

- `train_sensicut_fastai.py`: 主要的训练脚本
- `predict_fastai.py`: 预测脚本，用于使用训练好的模型进行推理
- `requirements_fastai.txt`: 依赖包列表
- `README_fastai.md`: 使用说明（本文件）

## 安装依赖

```bash
pip install -r requirements_fastai.txt
```

## 主要改进

### 相比原始 PyTorch 版本的优势：

1. **更简洁的代码**: fast.ai 提供了高级API，大大减少了代码量
2. **内置数据增强**: 使用 `aug_transforms()` 自动应用多种数据增强技术
3. **自动学习率调度**: 使用 `fit_one_cycle()` 实现一周期学习率调度
4. **分阶段训练**: 自动实现冻结-解冻的迁移学习策略
5. **内置可视化**: 提供丰富的可视化工具（损失图、混淆矩阵等）
6. **更好的模型解释**: 内置模型解释工具，帮助理解模型预测
7. **更智能的学习率调度**: 使用一周期学习率调度替代固定学习率

### 保持不变的参数：

- 批大小: 32 (适合 M2 芯片)
- 训练轮数: 20
- 学习率: 0.003
- 数据集分割: 80% 训练，20% 验证
- 模型架构: ResNet-50
- 优化器: Adam (显式设置以保持与原版一致)

## 使用方法

### 1. 训练模型

```bash
python train_sensicut_fastai.py
```

训练完成后会生成以下文件：
- `sensicut_resnet50_fastai.pkl`: 导出的模型文件（用于推理）
- `class_names_fastai.json`: 类别名称文件
- `training_loss_fastai.png`: 训练损失图
- `confusion_matrix_fastai.png`: 混淆矩阵
- `top_losses_fastai.png`: 预测错误最多的样本
- `sample_images_fastai.png`: 训练样本预览

### 2. 使用模型进行预测

```bash
python predict_fastai.py
```

或者在 Python 代码中使用：

```python
from predict_fastai import SensiCutPredictor

# 初始化预测器
predictor = SensiCutPredictor()

# 预测单张图片
result = predictor.predict_single_image("path/to/image.jpg", top_k=5)
print(f"预测结果: {result['predicted_class']}")
print(f"置信度: {result['confidence']:.3f}")

# 批量预测
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
results = predictor.predict_batch(image_paths)

# 评估测试文件夹
evaluation = predictor.evaluate_test_folder("test_images/")
```

## 优化器说明

**重要**: 原本 fast.ai 默认使用 SGD 优化器，但为了保持与原版一致，我们显式设置了 Adam 优化器：

```python
learner = vision_learner(
    dls,
    arch,
    metrics=[accuracy],
    pretrained=True,
    opt_func=Adam  # 使用Adam优化器以保持与原版一致
)
```

## 训练流程

fast.ai 版本采用了更先进的训练策略：

1. **第一阶段 (5 个 epoch)**: 
   - 冻结预训练的 ResNet-50 层
   - 只训练最后的分类头
   - 使用较高的学习率 (0.003)

2. **第二阶段 (15 个 epoch)**:
   - 解冻所有层进行微调
   - 使用较低的学习率 (0.0003)
   - 应用一周期学习率调度

## 数据增强

fast.ai 版本使用了更丰富的数据增强技术：

- 随机水平翻转
- 随机旋转 (±10°)
- 随机缩放 (0.8-1.2倍)
- 随机亮度和对比度调整 (±30%)
- 自动应用 ImageNet 标准化

## 模型评估

训练完成后，脚本会自动生成：

1. **训练历史图**: 显示训练和验证损失的变化
2. **混淆矩阵**: 显示各类别的预测准确性
3. **预测错误分析**: 显示预测错误最多的样本
4. **分类报告**: 每个类别的精确度、召回率和F1分数

## 设备支持

支持多种计算设备：
- ✅ NVIDIA GPU (CUDA)
- ✅ Apple Silicon GPU (MPS)
- ✅ CPU

## 性能优化

为了在不同设备上获得最佳性能：

- **M2 芯片**: 使用较小的批大小 (32) 和较少的 worker 线程 (2)
- **CUDA GPU**: 可以使用更大的批大小和更多的 worker 线程
- **CPU**: 自动降低批大小以适应内存限制

## 故障排除

### 常见问题：

1. **内存不足**: 
   - 减少批大小 (BATCH_SIZE)
   - 减少 worker 线程数量

2. **训练速度慢**:
   - 检查是否正确使用了 GPU 加速
   - 在 M2 芯片上确保使用了 MPS 后端

3. **模型加载失败**:
   - 确保 `sensicut_resnet50_fastai.pkl` 文件存在
   - 检查 fast.ai 版本是否兼容

### 调试技巧：

```python
# 查看数据加载器
dls.show_batch()

# 检查模型架构
learner.model

# 查看学习率建议
learner.lr_find()

# 查看训练记录
learner.recorder.plot_loss()
```

## 进阶使用

### 自定义数据增强：

```python
# 在 create_dataloaders 函数中修改
batch_tfms=[
    *aug_transforms(
        do_flip=True,
        flip_vert=False,
        max_rotate=15.0,  # 增加旋转角度
        max_zoom=1.5,     # 增加缩放范围
        max_lighting=0.4, # 增加光照变化
        # 添加更多增强...
    ),
    Normalize.from_stats(*imagenet_stats)
]
```

### 使用不同的模型架构：

```python
# 在 create_model 函数中修改
learner = vision_learner(
    dls,
    resnet101,  # 使用更大的模型
    metrics=[accuracy],
    pretrained=True
)
```

## 总结

fast.ai 版本相比原始 PyTorch 版本提供了：

- 📦 更简洁的代码结构
- 🚀 更快的开发速度
- 📊 更丰富的可视化工具
- 🔧 更智能的默认设置
- 🎯 更好的训练策略

适合快速原型开发和实验，同时保持了与原版本相同的核心功能和性能。 