import os
import torch
import torchvision.transforms as T
from PIL import Image
import random
import matplotlib.pyplot as plt

# 设置输入输出目录
input_dir = '/root/autodl-tmp/Semi-supervised-learning/notebooks/cheat_visualization_input'  # 输入图像目录
output_dir = '/root/autodl-tmp/Semi-supervised-learning/notebooks/cheat_stronglyaug_output'  # 输出图像目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 强增强的 Transform
strong_transform = T.Compose([
    T.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # 随机裁剪
    T.RandomHorizontalFlip(),  # 随机水平翻转
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # 随机颜色变化
    T.RandomRotation(degrees=30),  # 随机旋转
    T.ToTensor(),  # 转为 Tensor
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 标准化
])

# 从文件夹中加载图像并应用增强
def load_and_augment_image(image_path, output_path):
    image = Image.open(image_path).convert('RGB')  # 确保是 RGB 格式
    augmented_image = strong_transform(image)  # 应用增强
    
    # 保存增强后的图像
    augmented_image = augmented_image.permute(1, 2, 0).numpy()  # 转换回 (H, W, C) 格式以便显示
    augmented_image = (augmented_image * 255).astype('uint8')  # 反标准化
    augmented_pil = Image.fromarray(augmented_image)  # 转回 PIL 格式
    augmented_pil.save(output_path)  # 保存到输出目录

# 批量处理输入目录中的所有图片
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    if os.path.isfile(image_path):
        print(f"Processing {image_name}...")
        output_path = os.path.join(output_dir, f"aug_{image_name}")  # 为增强图像添加前缀
        load_and_augment_image(image_path, output_path)

print("Augmentation complete.")
