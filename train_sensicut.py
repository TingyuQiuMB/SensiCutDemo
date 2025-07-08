import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

class SensiCutDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

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
            labels.append(i)
    
    return image_paths, labels, class_names

def create_model(num_classes):
    """创建ResNet-50模型用于迁移学习"""
    # 加载预训练的ResNet-50模型
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    
    # 冻结所有层（迁移学习）
    for param in model.parameters():
        param.requires_grad = False
    
    # 替换最后的分类层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

def train_model(model, train_loader, val_loader, num_epochs, device):
    """训练模型"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        train_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct_predictions / total_predictions:.2f}%'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct_predictions / total_predictions
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total_predictions += labels.size(0)
                val_correct_predictions += (predicted == labels).sum().item()
                
                val_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * val_correct_predictions / val_total_predictions:.2f}%'
                })
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = 100.0 * val_correct_predictions / val_total_predictions
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, train_accuracies, val_accuracies

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失图
    ax1.plot(train_losses, label='训练损失')
    ax1.plot(val_losses, label='验证损失')
    ax1.set_title('损失变化')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率图
    ax2.plot(train_accuracies, label='训练准确率')
    ax2.plot(val_accuracies, label='验证准确率')
    ax2.set_title('准确率变化')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def main():
    # 设置参数
    DATA_DIR = 'SensiCut2021'
    # BATCH_SIZE = 64
    BATCH_SIZE = 32  # M2芯片建议使用较小的batch size
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.003
    
    # 检查设备 (支持Apple M1/M2芯片的MPS加速)
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
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.2)),  # 随机缩放±20%
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 亮度和对比度随机调整±30%
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据
    print("正在加载数据...")
    image_paths, labels, class_names = load_data(DATA_DIR)
    print(f"总共加载了 {len(image_paths)} 张图片")
    print(f"类别数量: {len(class_names)} (注意：实际是59种材料，不是30种)")
    
    # 分割数据集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # 创建数据集和数据加载器
    train_dataset = SensiCutDataset(train_paths, train_labels, transform=transform_train)
    val_dataset = SensiCutDataset(val_paths, val_labels, transform=transform_val)
    
    # 调整num_workers以适应M2芯片
    num_workers = 2 if device.type == 'mps' else 4
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    model = create_model(len(class_names))
    model = model.to(device)
    
    print(f"模型已加载到 {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # 训练模型
    print("\n开始训练...")
    model, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, NUM_EPOCHS, device
    )
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'num_classes': len(class_names)
    }, 'sensicut_resnet50_model.pth')
    
    # 保存类别名称
    with open('class_names.json', 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 最终评估
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算最终准确率
    final_accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n最终验证准确率: {final_accuracy * 100:.2f}%")
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    print(f"\n模型已保存为 'sensicut_resnet50_model.pth'")
    print(f"类别名称已保存为 'class_names.json'")
    print(f"训练历史图已保存为 'training_history.png'")

if __name__ == "__main__":
    main() 