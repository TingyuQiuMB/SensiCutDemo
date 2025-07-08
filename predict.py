import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
import argparse
import os

def load_model(model_path, device):
    """加载训练好的模型"""
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 创建模型
    model = models.resnet50(pretrained=False)
    num_classes = checkpoint['num_classes']
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint['class_names']

def preprocess_image(image_path):
    """预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # 添加batch维度
    return image

def predict(model, image_tensor, class_names, device, top_k=5):
    """预测图像类别"""
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # 获取top-k预测
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        results = []
        for i in range(top_k):
            class_idx = top_indices[i].item()
            class_name = class_names[class_idx]
            probability = top_probs[i].item()
            results.append((class_name, probability))
    
    return results

def main():
    parser = argparse.ArgumentParser(description='SensiCut材料分类预测')
    parser.add_argument('--image', type=str, required=True, help='要预测的图像路径')
    parser.add_argument('--model', type=str, default='sensicut_resnet50_model.pth', 
                       help='模型文件路径')
    parser.add_argument('--top_k', type=int, default=5, help='显示前k个预测结果')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.image):
        print(f"错误：图像文件 {args.image} 不存在")
        return
    
    if not os.path.exists(args.model):
        print(f"错误：模型文件 {args.model} 不存在")
        return
    
    # 设备检查
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print("正在加载模型...")
    model, class_names = load_model(args.model, device)
    print(f"模型加载成功，支持 {len(class_names)} 种材料分类")
    
    # 预处理图像
    print(f"正在预处理图像: {args.image}")
    image_tensor = preprocess_image(args.image)
    
    # 预测
    print("正在进行预测...")
    results = predict(model, image_tensor, class_names, device, args.top_k)
    
    # 显示结果
    print(f"\n预测结果 (Top {args.top_k}):")
    print("-" * 50)
    for i, (class_name, probability) in enumerate(results):
        print(f"{i+1}. {class_name}: {probability:.4f} ({probability*100:.2f}%)")

if __name__ == "__main__":
    main() 