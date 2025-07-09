import torch
import torch.nn as nn
import fastai
from fastai.vision.all import *
from fastai.metrics import accuracy
from fastai.optimizer import Adam
import os
import json
import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path

def load_data(data_dir):
    """加载SensiCut2021数据集"""
    image_paths = []
    labels = []
    class_names = []
    
    # 获取所有材料类型文件夹
    material_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    material_folders.sort()
    
    print(f"发现 {len(material_folders)} 种材料类型:")
    
    for i, folder in enumerate(material_folders):
        # 提取材料名称（去掉日期时间前缀）
        material_name = folder.split('-', 3)[-1] if '-' in folder else folder
        class_names.append(material_name)
        print(f"{i+1}. {material_name}")
        
        folder_path = os.path.join(data_dir, folder)
        
        # 获取该材料的所有图片文件
        image_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        
        for img_path in image_files:
            image_paths.append(img_path)
            labels.append(material_name)  # 使用材料名称作为标签
    
    return image_paths, labels, class_names

def create_dataframe(image_paths, labels):
    """创建DataFrame用于fast.ai"""
    import pandas as pd
    
    df = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })
    
    return df

def create_dataloaders(df, batch_size=32, valid_pct=0.2):
    """创建fast.ai数据加载器"""
    
    # 创建DataBlock
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_x=lambda x: x['image_path'],
        get_y=lambda x: x['label'],
        splitter=RandomSplitter(valid_pct=valid_pct, seed=42),
        item_tfms=[
            Resize(256, method='squish')
        ],
        batch_tfms=[
            *aug_transforms(
                do_flip=True,
                flip_vert=False,
                max_rotate=10.0,
                max_zoom=1.2,
                min_zoom=0.8,
                max_lighting=0.3,
                max_warp=0.0,
                p_affine=0.5,
                p_lighting=0.5
            ),
            Normalize.from_stats(*imagenet_stats)
        ]
    )
    
    # 创建DataLoaders
    dls = dblock.dataloaders(df, bs=batch_size)
    
    return dls

def create_model(dls, arch=resnet50):
    """创建fast.ai模型"""
    
    # 创建视觉学习器，使用预训练的ResNet-50
    learner = vision_learner(
        dls,
        arch,
        metrics=[accuracy],
        pretrained=True,
        opt_func=Adam  # 使用Adam优化器以保持与原版一致
    )
    
    return learner

def train_model(learner, num_epochs=20, lr=3e-3):
    """训练模型"""
    
    print("开始训练...")
    
    # 冻结预训练层，只训练分类头
    print("第一阶段：冻结预训练层，训练分类头")
    learner.freeze()
    
    # 训练分类头（使用Adam优化器 + 一周期调度）
    learner.fit_one_cycle(5, lr)
    
    # 解冻所有层进行微调
    print("第二阶段：解冻所有层进行微调")
    learner.unfreeze()
    
    # 使用较小的学习率进行微调（使用Adam优化器 + 一周期调度）
    learner.fit_one_cycle(num_epochs-5, lr/10)
    
    return learner

def plot_training_history(learner):
    """绘制训练历史"""
    
    # 绘制损失图
    learner.recorder.plot_loss()
    plt.title('训练和验证损失')
    plt.savefig('training_loss_fastai.png')
    plt.show()

def evaluate_model(learner, class_names):
    """评估模型"""
    
    # 获取验证集的预测结果
    interp = ClassificationInterpretation.from_learner(learner)
    
    # 显示混淆矩阵
    interp.plot_confusion_matrix(figsize=(12, 12))
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix_fastai.png')
    plt.show()
    
    # 显示预测错误最多的类别
    interp.plot_top_losses(9, figsize=(15, 10))
    plt.title('预测错误最多的样本')
    plt.savefig('top_losses_fastai.png')
    plt.show()
    
    # 计算每个类别的准确率
    val_preds, val_targets = learner.get_preds()
    val_preds = torch.argmax(val_preds, dim=1)
    
    # 转换为numpy数组
    val_preds_np = val_preds.cpu().numpy()
    val_targets_np = val_targets.cpu().numpy()
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(val_targets_np, val_preds_np, target_names=class_names))
    
    # 计算最终准确率
    final_accuracy = accuracy_score(val_targets_np, val_preds_np)
    print(f"\n最终验证准确率: {final_accuracy * 100:.2f}%")
    
    return final_accuracy

def main():
    # 设置参数（保持与原文件相同）
    DATA_DIR = 'SensiCut2021'
    BATCH_SIZE = 32  # M2芯片建议使用较小的batch size
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.003
    
    # 检查设备支持
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'使用设备: {device} (NVIDIA GPU)')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f'使用设备: {device} (Apple Silicon GPU)')
    else:
        device = torch.device('cpu')
        print(f'使用设备: {device} (CPU)')
    
    print(f'PyTorch版本: {torch.__version__}')
    print(f'fast.ai版本: {fastai.__version__}')
    
    # 加载数据
    print("正在加载数据...")
    image_paths, labels, class_names = load_data(DATA_DIR)
    print(f"总共加载了 {len(image_paths)} 张图片")
    print(f"类别数量: {len(class_names)} (注意：实际是59种材料，不是30种)")
    
    # 创建DataFrame
    df = create_dataframe(image_paths, labels)
    print(f"数据框创建完成，形状: {df.shape}")
    
    # 创建数据加载器
    print("创建数据加载器...")
    dls = create_dataloaders(df, batch_size=BATCH_SIZE, valid_pct=0.2)
    
    print(f"训练集大小: {len(dls.train_ds)}")
    print(f"验证集大小: {len(dls.valid_ds)}")
    print(f"类别数量: {len(dls.vocab)}")
    
    # 显示一些样本
    dls.show_batch(max_n=9, figsize=(12, 8))
    plt.title('训练样本预览')
    plt.savefig('sample_images_fastai.png')
    plt.show()
    
    # 创建模型
    print("创建模型...")
    learner = create_model(dls, arch=resnet50)
    
    # 显示模型架构
    print(f"模型架构: {learner.model.__class__.__name__}")
    
    # 训练模型
    trained_learner = train_model(learner, NUM_EPOCHS, LEARNING_RATE)
    
    # 保存模型
    print("保存模型...")
    trained_learner.save('sensicut_resnet50_fastai')
    
    # 导出模型（用于推理）
    trained_learner.export('sensicut_resnet50_fastai.pkl')
    
    # 保存类别名称
    with open('class_names_fastai.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    
    # 绘制训练历史
    print("绘制训练历史...")
    plot_training_history(trained_learner)
    
    # 评估模型
    print("评估模型...")
    final_accuracy = evaluate_model(trained_learner, class_names)
    
    print(f"\n训练完成！")
    print(f"模型已保存为 'sensicut_resnet50_fastai.pkl'")
    print(f"类别名称已保存为 'class_names_fastai.json'")
    print(f"训练历史图已保存为 'training_loss_fastai.png'")
    print(f"最终验证准确率: {final_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main() 