import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import platform

# 配置中文字体
def setup_chinese_font():
    """设置中文字体"""
    system = platform.system()
    try:
        if system == "Darwin":  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei']
        elif system == "Windows":
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
        elif system == "Linux":
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
        
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        return True
    except:
        print("警告: 无法设置中文字体，将使用英文标题")
        return False

def create_augmentation_transforms():
    """创建数据增强变换"""
    transform_augmented = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.8, 1.2)),  # 随机缩放±20%
        transforms.ColorJitter(brightness=0.3, contrast=0.3),  # 亮度和对比度随机调整±30%
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    
    transform_original = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    return transform_augmented, transform_original

def tensor_to_image(tensor):
    """将tensor转换为可显示的图像"""
    tensor = tensor.clone()
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.permute(1, 2, 0).numpy()

def get_labels(use_chinese=True):
    """获取标签文本"""
    if use_chinese:
        return {
            'original': '原始图像',
            'augmented': '增强样本',
            'title': '数据增强效果展示',
            'saved': '数据增强示例已保存为'
        }
    else:
        return {
            'original': 'Original Image',
            'augmented': 'Augmented Sample',
            'title': 'Data Augmentation Examples',
            'saved': 'Data augmentation examples saved as'
        }

def show_augmentation_examples(image_path, num_examples=8, use_chinese=True):
    """展示数据增强的效果"""
    if not os.path.exists(image_path):
        print(f"错误：图像文件 {image_path} 不存在")
        return
    
    labels = get_labels(use_chinese)
    
    # 创建变换
    transform_augmented, transform_original = create_augmentation_transforms()
    
    # 加载原始图像
    original_image = Image.open(image_path).convert('RGB')
    original_tensor = transform_original(original_image)
    
    # 创建子图
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    
    # 显示原始图像
    axes[0, 0].imshow(tensor_to_image(original_tensor))
    axes[0, 0].set_title(labels['original'], fontsize=12)
    axes[0, 0].axis('off')
    
    # 生成增强后的图像
    for i in range(num_examples):
        row = (i + 1) // 3
        col = (i + 1) % 3
        
        # 应用数据增强
        augmented_tensor = transform_augmented(original_image)
        
        # 显示增强后的图像
        axes[row, col].imshow(tensor_to_image(augmented_tensor))
        axes[row, col].set_title(f'{labels["augmented"]} {i+1}', fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.suptitle(labels['title'], fontsize=16, y=0.98)
    plt.savefig('data_augmentation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"{labels['saved']} 'data_augmentation_examples.png'")

def analyze_augmentation_distribution(image_path, num_samples=100, use_chinese=True):
    """分析数据增强的分布"""
    if not os.path.exists(image_path):
        print(f"错误：图像文件 {image_path} 不存在")
        return
    
    # 标签文本
    if use_chinese:
        labels = {
            'xlabel': '平均亮度值',
            'ylabel': '频数',
            'title': '数据增强后图像亮度分布',
            'mean_label': '平均值',
            'saved': '亮度分布分析已保存为',
            'stats': '亮度统计:',
            'mean': '平均值',
            'std': '标准差',
            'min': '最小值',
            'max': '最大值'
        }
    else:
        labels = {
            'xlabel': 'Average Brightness',
            'ylabel': 'Frequency',
            'title': 'Image Brightness Distribution after Data Augmentation',
            'mean_label': 'Mean',
            'saved': 'Brightness distribution analysis saved as',
            'stats': 'Brightness Statistics:',
            'mean': 'Mean',
            'std': 'Std Dev',
            'min': 'Min',
            'max': 'Max'
        }
    
    transform_augmented, _ = create_augmentation_transforms()
    original_image = Image.open(image_path).convert('RGB')
    
    # 生成多个增强样本
    brightness_values = []
    
    for _ in range(num_samples):
        # 创建一个专门用于分析亮度的变换
        brightness_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
        ])
        
        augmented_tensor = brightness_transform(original_image)
        brightness = torch.mean(augmented_tensor).item()
        brightness_values.append(brightness)
    
    # 绘制亮度分布
    plt.figure(figsize=(10, 6))
    plt.hist(brightness_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel(labels['xlabel'])
    plt.ylabel(labels['ylabel'])
    plt.title(labels['title'])
    plt.grid(True, alpha=0.3)
    plt.axvline(x=np.mean(brightness_values), color='red', linestyle='--', 
                label=f'{labels["mean_label"]}: {np.mean(brightness_values):.3f}')
    plt.legend()
    plt.savefig('brightness_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"{labels['saved']} 'brightness_distribution.png'")
    print(f"{labels['stats']}")
    print(f"  {labels['mean']}: {np.mean(brightness_values):.3f}")
    print(f"  {labels['std']}: {np.std(brightness_values):.3f}")
    print(f"  {labels['min']}: {min(brightness_values):.3f}")
    print(f"  {labels['max']}: {max(brightness_values):.3f}")

def main():
    parser = argparse.ArgumentParser(description='测试数据增强效果')
    parser.add_argument('--image', type=str, required=True, help='测试图像路径')
    parser.add_argument('--samples', type=int, default=8, help='生成的增强样本数量')
    parser.add_argument('--analysis', action='store_true', help='进行分布分析')
    parser.add_argument('--english', action='store_true', help='使用英文标题')
    
    args = parser.parse_args()
    
    # 设置字体
    chinese_font_available = setup_chinese_font() and not args.english
    
    print("=" * 50)
    if chinese_font_available:
        print("SensiCut 数据增强测试")
    else:
        print("SensiCut Data Augmentation Test")
    print("=" * 50)
    
    # 显示数据增强示例
    if chinese_font_available:
        print("\n1. 生成数据增强示例...")
    else:
        print("\n1. Generating data augmentation examples...")
    show_augmentation_examples(args.image, args.samples, use_chinese=chinese_font_available)
    
    # 可选：进行分布分析
    if args.analysis:
        if chinese_font_available:
            print("\n2. 进行亮度分布分析...")
        else:
            print("\n2. Analyzing brightness distribution...")
        analyze_augmentation_distribution(args.image, 100, use_chinese=chinese_font_available)
    
    if chinese_font_available:
        print("\n数据增强包含以下变换:")
        print("- 随机缩放裁剪: 0.8-1.2倍 (±20%)")
        print("- 亮度调整: ±30%")
        print("- 对比度调整: ±30%")
        print("- 随机水平翻转")
        print("- 随机旋转: ±10度")
    else:
        print("\nData augmentation includes the following transforms:")
        print("- Random resized crop: 0.8-1.2x scale (±20%)")
        print("- Brightness adjustment: ±30%")
        print("- Contrast adjustment: ±30%")
        print("- Random horizontal flip")
        print("- Random rotation: ±10 degrees")

if __name__ == "__main__":
    main() 