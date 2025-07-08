# SensiCut 激光切割材料分类 - ResNet-50 迁移学习

这个项目使用ResNet-50预训练模型对SensiCut2021数据集中的激光切割材料进行分类。

## 数据集信息

- **数据集名称**: SensiCut2021
- **材料类型数量**: 59种（不是30种）
- **数据格式**: JPG图像文件
- **材料类型**: 包括各种木材、亚克力、金属、织物等激光切割材料

## 环境要求

- Python 3.7+
- PyTorch 1.9.0+
- CUDA支持（可选，但推荐用于GPU加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

1. 确保SensiCut2021数据集在项目根目录下
2. 运行训练脚本：

```bash
python train_sensicut.py
```

### 预测新图像

训练完成后，可以使用训练好的模型对新图像进行预测：

```bash
python predict.py --image path/to/your/image.jpg
```

可选参数：
- `--model`: 指定模型文件路径（默认：sensicut_resnet50_model.pth）
- `--top_k`: 显示前k个预测结果（默认：5）

示例：
```bash
python predict.py --image test_image.jpg --top_k 3
```

### 测试数据增强效果

在训练前，您可以使用测试脚本查看数据增强的效果：

```bash
python test_data_augmentation.py --image path/to/test/image.jpg
```

可选参数：
- `--samples`: 生成的增强样本数量（默认：8）
- `--analysis`: 进行亮度分布分析
- `--english`: 使用英文标题（解决中文字体问题）

示例：
```bash
# 使用中文标题（自动检测字体）
python test_data_augmentation.py --image SensiCut2021/2020.07.17-14.54.51-RedOakHardwood/RedOakHardwood-i0-r0-z56.2-x172.0-y103.0.jpg --analysis

# 使用英文标题（避免字体问题）
python test_data_augmentation.py --image SensiCut2021/2020.07.17-14.54.51-RedOakHardwood/RedOakHardwood-i0-r0-z56.2-x172.0-y103.0.jpg --analysis --english
```

## 训练参数

- **预训练模型**: ResNet-50 (ImageNet预训练)
- **优化器**: Adam
- **学习率**: 0.003
- **批量大小**: 64
- **训练轮数**: 20
- **图像尺寸**: 256×256
- **数据增强**: 
  - 随机缩放±20%
  - 亮度和对比度随机调整±30%
  - 随机水平翻转和旋转

## 数据预处理

### 训练集数据增强
- 随机缩放裁剪至256×256（缩放范围：0.8-1.2倍，即±20%）
- 亮度和对比度随机调整（±30%）
- 随机水平翻转
- 随机旋转（±10度）
- ImageNet标准化

### 验证集预处理
- 图像大小调整为256×256
- ImageNet标准化

### 数据增强目的
数据增强旨在：
- **提高模型泛化能力**: 通过增加训练数据的多样性
- **模拟真实环境**: 随机缩放和光照变化模拟不同的拍摄条件
- **增强鲁棒性**: 使模型对不同光照条件和视角变化更加鲁棒

## 输出文件

训练完成后会生成以下文件：

- `sensicut_resnet50_model.pth` - 训练好的模型（包含模型权重和类别信息）
- `class_names.json` - 类别名称映射文件
- `training_history.png` - 训练历史图表（损失和准确率曲线）

## 项目文件结构

```
SensiCut/
├── train_sensicut.py              # 训练脚本
├── predict.py                     # 推理脚本
├── test_data_augmentation.py      # 数据增强测试脚本
├── requirements.txt               # 依赖包列表
├── README.md                      # 说明文档
├── SensiCut2021/                 # 数据集目录
│   ├── 2020.07.17-14.54.51-RedOakHardwood/
│   ├── 2020.07.17-15.50.11-MapleHardWood/
│   └── ...
├── sensicut_resnet50_model.pth        # 训练好的模型（训练后生成）
├── class_names.json                   # 类别映射（训练后生成）
├── training_history.png               # 训练历史图表（训练后生成）
├── data_augmentation_examples.png     # 数据增强示例（测试后生成）
└── brightness_distribution.png        # 亮度分布分析（测试后生成）
```

## 模型架构

- 使用预训练的ResNet-50作为特征提取器
- 冻结所有卷积层（迁移学习）
- 替换最后的分类层以适应59个类别

## 注意事项

1. **类别数量**: 实际数据集包含59种材料，不是30种
2. **GPU内存**: 建议使用至少8GB显存的GPU
3. **训练时间**: 在GPU上大约需要1-2小时
4. **数据集大小**: 确保有足够的磁盘空间存储数据集
5. **数据增强**: 加强的数据增强可能会：
   - 延长训练时间（约10-15%）
   - 提高模型的泛化能力
   - 在训练初期可能导致损失收敛较慢

## 材料类型列表

数据集包含以下59种激光切割材料（按字母顺序）：

1. ABS - ABS塑料
2. Acetate - 醋酸纤维
3. AmberBamboo - 琥珀色竹子
4. BambooPremiumVeneerMDF - 竹子高级贴面MDF
5. BirchPlywood - 桦木胶合板
6. BlackAcrylic - 黑色亚克力
7. BlackCardstock - 黑色卡纸
8. BlackCoatedMDF - 黑色涂层MDF
9. BlackDelrin - 黑色德林
10. BlackFelt - 黑色毡子
11. BlackLeather - 黑色皮革
12. BlackMatboard - 黑色卡纸板
13. BlackMatteAcrylic - 黑色磨砂亚克力
14. BlackMelamineMDF - 黑色三聚氰胺MDF
15. BlackSilicone - 黑色硅胶
16. BlackSuede - 黑色绒面革
17. BlondeBamboo - 金色竹子
18. BrownCardboard - 棕色纸板
19. BrownCorrugatedCard - 棕色瓦楞纸
20. CarbonFiber - 碳纤维
21. CarbonSteel - 碳钢
22. ClearAcrylic - 透明亚克力
23. ClearMatteAcrylic - 透明磨砂亚克力
24. Cork - 软木
25. CreamAcrylic - 奶油色亚克力
26. ExtrudedAcrylicBlack - 黑色挤出亚克力
27. ExtrudedAcrylicClear - 透明挤出亚克力
28. FireWoolFelt - 火焰色羊毛毡
29. Foamboard - 泡沫板
30. GreenAcrylic - 绿色亚克力
31. GreenCardstock - 绿色卡纸
32. GreenMatboard - 绿色卡纸板
33. GreenTintedAcrylic - 绿色有色亚克力
34. GreyCardstock - 灰色卡纸
35. IvoryCardstock - 象牙色卡纸
36. Lexan - 聚碳酸酯
37. LexanWhite - 白色聚碳酸酯
38. MapleHardWood - 枫木硬木
39. MDFnatural - 天然MDF
40. OrangeAcrylic - 橙色亚克力
41. OrangeWoolFelt - 橙色羊毛毡
42. PETG - PETG塑料
43. PVCClear - 透明PVC
44. RedAcrylic - 红色亚克力
45. RedOakHardwood - 红橡木硬木
46. RedPaper - 红色纸
47. RedSuede - 红色绒面革
48. RedSyntheticFelt - 红色合成毡
49. StainlessSteel - 不锈钢
50. StandardAluminum - 标准铝
51. WalnutHardWood - 胡桃木硬木
52. WhiteAcrylic - 白色亚克力
53. WhiteCoasterboard - 白色杯垫纸板
54. WhiteCorrugatedCard - 白色瓦楞纸
55. WhiteDelrin - 白色德林
56. WhiteFelt - 白色毡子
57. WhiteMatteAcrylic - 白色磨砂亚克力
58. WhiteMelamineMDF - 白色三聚氰胺MDF
59. WhiteStyrene - 白色苯乙烯

注意：数据集确实包含59种不同的激光切割材料。

## 模型评估

训练过程中会自动进行验证，并在训练结束后生成详细的分类报告。

## 故障排除

1. **内存不足**: 减少batch_size或使用更小的图像尺寸
2. **CUDA错误**: 确保PyTorch与CUDA版本兼容
3. **数据加载错误**: 检查数据集路径是否正确
4. **训练收敛慢**: 数据增强可能导致训练初期收敛较慢，这是正常现象
5. **过拟合**: 如果验证准确率远低于训练准确率，可以：
   - 增加更多数据增强
   - 降低学习率
   - 增加正则化
6. **中文字体问题**: 如果数据增强测试脚本出现字体警告：
   - 方案1：使用 `--english` 参数：`python test_data_augmentation.py --image xxx.jpg --english`
   - 方案2：安装中文字体（macOS系统自带PingFang SC字体应该自动检测）
   - 方案3：手动安装matplotlib中文字体包：`pip install matplotlib -U` # SensiCutDemo
