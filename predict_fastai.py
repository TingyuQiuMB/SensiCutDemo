import torch
from fastai.vision.all import *
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os
import glob

class SensiCutPredictor:
    def __init__(self, model_path='sensicut_resnet50_fastai.pkl', class_names_path='class_names_fastai.json'):
        """
        初始化预测器
        
        Args:
            model_path: 导出的fast.ai模型路径
            class_names_path: 类别名称文件路径
        """
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.learner = None
        self.class_names = None
        
        # 加载模型和类别名称
        self.load_model()
        self.load_class_names()
    
    def load_model(self):
        """加载fast.ai模型"""
        try:
            self.learner = load_learner(self.model_path)
            print(f"模型加载成功: {self.model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def load_class_names(self):
        """加载类别名称"""
        try:
            with open(self.class_names_path, 'r', encoding='utf-8') as f:
                self.class_names = json.load(f)
            print(f"类别名称加载成功: {len(self.class_names)} 个类别")
        except Exception as e:
            print(f"类别名称加载失败: {e}")
            # 如果加载失败，使用模型中的词汇表
            if self.learner:
                self.class_names = self.learner.dls.vocab
                print(f"使用模型词汇表: {len(self.class_names)} 个类别")
    
    def predict_single_image(self, image_path, show_image=True, top_k=3):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            show_image: 是否显示图片
            top_k: 返回前k个预测结果
        
        Returns:
            dict: 预测结果
        """
        try:
            # 加载图片
            image = Image.open(image_path).convert('RGB')
            
            # 使用模型预测
            pred, pred_idx, probs = self.learner.predict(image_path)
            
            # 获取前k个预测结果
            top_k_indices = torch.topk(probs, top_k).indices
            top_k_probs = torch.topk(probs, top_k).values
            
            results = {
                'predicted_class': str(pred),
                'confidence': float(probs[pred_idx]),
                'top_k_predictions': []
            }
            
            for i in range(top_k):
                idx = top_k_indices[i]
                prob = top_k_probs[i]
                class_name = self.learner.dls.vocab[idx]
                results['top_k_predictions'].append({
                    'class': class_name,
                    'confidence': float(prob)
                })
            
            # 显示图片和预测结果
            if show_image:
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title(f'输入图片\n{Path(image_path).name}')
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                classes = [item['class'] for item in results['top_k_predictions']]
                confidences = [item['confidence'] for item in results['top_k_predictions']]
                
                bars = plt.barh(range(len(classes)), confidences)
                plt.yticks(range(len(classes)), classes)
                plt.xlabel('置信度')
                plt.title(f'前{top_k}个预测结果')
                
                # 为每个条形添加数值标签
                for i, (bar, conf) in enumerate(zip(bars, confidences)):
                    plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                            f'{conf:.3f}', ha='left', va='center')
                
                plt.tight_layout()
                plt.show()
            
            return results
            
        except Exception as e:
            print(f"预测失败: {e}")
            return None
    
    def predict_batch(self, image_paths, show_results=True):
        """
        批量预测多张图片
        
        Args:
            image_paths: 图片路径列表
            show_results: 是否显示结果
        
        Returns:
            list: 预测结果列表
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict_single_image(image_path, show_image=False)
            if result:
                result['image_path'] = image_path
                results.append(result)
        
        if show_results and results:
            self.show_batch_results(results)
        
        return results
    
    def show_batch_results(self, results, max_images=9):
        """显示批量预测结果"""
        n_images = min(len(results), max_images)
        cols = 3
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        for i in range(n_images):
            result = results[i]
            image_path = result['image_path']
            
            # 加载并显示图片
            image = Image.open(image_path).convert('RGB')
            axes[i].imshow(image)
            
            # 设置标题
            title = f"{Path(image_path).name}\n"
            title += f"预测: {result['predicted_class']}\n"
            title += f"置信度: {result['confidence']:.3f}"
            
            axes[i].set_title(title, fontsize=10)
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_test_folder(self, test_folder, show_summary=True):
        """
        评估测试文件夹中的所有图片
        
        Args:
            test_folder: 测试文件夹路径
            show_summary: 是否显示结果摘要
        
        Returns:
            dict: 评估结果
        """
        # 获取所有图片文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(test_folder, ext)))
        
        if not image_paths:
            print(f"在 {test_folder} 中没有找到图片文件")
            return None
        
        print(f"找到 {len(image_paths)} 张图片")
        
        # 批量预测
        results = self.predict_batch(image_paths, show_results=False)
        
        # 统计结果
        class_counts = {}
        total_confidence = 0
        
        for result in results:
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            if predicted_class not in class_counts:
                class_counts[predicted_class] = []
            class_counts[predicted_class].append(confidence)
            total_confidence += confidence
        
        # 计算统计信息
        stats = {
            'total_images': len(results),
            'average_confidence': total_confidence / len(results),
            'class_distribution': {}
        }
        
        for class_name, confidences in class_counts.items():
            stats['class_distribution'][class_name] = {
                'count': len(confidences),
                'percentage': len(confidences) / len(results) * 100,
                'avg_confidence': np.mean(confidences),
                'min_confidence': np.min(confidences),
                'max_confidence': np.max(confidences)
            }
        
        if show_summary:
            self.show_evaluation_summary(stats)
        
        return {
            'stats': stats,
            'detailed_results': results
        }
    
    def show_evaluation_summary(self, stats):
        """显示评估摘要"""
        print("\n=== 评估结果摘要 ===")
        print(f"总图片数: {stats['total_images']}")
        print(f"平均置信度: {stats['average_confidence']:.3f}")
        print(f"\n各类别分布:")
        print(f"{'类别':<20} {'数量':<8} {'百分比':<10} {'平均置信度':<12} {'最低置信度':<12} {'最高置信度'}")
        print("-" * 80)
        
        for class_name, info in stats['class_distribution'].items():
            print(f"{class_name:<20} {info['count']:<8} {info['percentage']:<10.1f}% "
                  f"{info['avg_confidence']:<12.3f} {info['min_confidence']:<12.3f} {info['max_confidence']:<12.3f}")

def main():
    """主函数 - 示例用法"""
    print("SensiCut Fast.ai 预测器")
    print("=" * 50)
    
    # 初始化预测器
    try:
        predictor = SensiCutPredictor()
    except Exception as e:
        print(f"初始化预测器失败: {e}")
        print("请确保已经训练了模型并生成了 sensicut_resnet50_fastai.pkl 文件")
        return
    
    # 示例1: 预测单张图片
    print("\n1. 单张图片预测示例")
    print("-" * 30)
    
    # 这里需要替换为实际的图片路径
    sample_image = "sample_image.jpg"  # 替换为实际的图片路径
    
    if os.path.exists(sample_image):
        result = predictor.predict_single_image(sample_image, show_image=True, top_k=5)
        if result:
            print(f"预测结果: {result['predicted_class']}")
            print(f"置信度: {result['confidence']:.3f}")
    else:
        print(f"示例图片 {sample_image} 不存在")
    
    # 示例2: 批量预测
    print("\n2. 批量预测示例")
    print("-" * 30)
    
    # 如果有测试文件夹，可以进行批量预测
    test_folder = "test_images"  # 替换为实际的测试文件夹路径
    
    if os.path.exists(test_folder):
        evaluation_result = predictor.evaluate_test_folder(test_folder, show_summary=True)
        
        if evaluation_result:
            print("\n详细预测结果已保存到 evaluation_result 变量中")
    else:
        print(f"测试文件夹 {test_folder} 不存在")
    
    print("\n预测完成!")

if __name__ == "__main__":
    main() 