import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import torchvision.transforms as T

# 设置输入输出目录
input_dir = '/root/autodl-tmp/Semi-supervised-learning/notebooks/cheat_visualization_input'  # 你的输入图片目录
output_dir = '/root/autodl-tmp/Semi-supervised-learning/notebooks/cheat_visualization_output'  # 你的输出图像目录

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 超参数
assume_imagenet_norm = True
blue_k = 8.0
blue_t = -0.035  # 可以调整阈值

# 图片反标准化函数
def denorm_if_needed(x, assume_imagenet_norm=True):
    if not assume_imagenet_norm:
        return x.clamp(0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
    return (x * std + mean).clamp(0, 1)

# 计算蓝色分数
@torch.no_grad()
def blue_score_map(x01):
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    return b - 0.5 * (r + g)

# 从批次中提取图片张量
def pick_first_image_tensor(batch):
    if isinstance(batch, dict):
        for k in ['x','image','img','x_lb','x_ulb_w','x_ulb_s']:
            if k in batch and isinstance(batch[k], torch.Tensor) and batch[k].dim() in (3,4):
                return batch[k]
        for _, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() in (3,4):
                return v
    elif isinstance(batch, (list, tuple)):
        for v in batch:
            if isinstance(v, torch.Tensor) and v.dim() in (3,4):
                return v
    return batch

# 创建软掩码
def soft_mask(score, blue_k, blue_t):
    return torch.sigmoid(blue_k * (score - blue_t))  # [B, 1, H, W]

# 从文件夹中加载图片
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保是 RGB 格式
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform(image).unsqueeze(0)  # Add batch dimension

# 可视化并保存结果
def visualize_and_save(image_tensor, output_path):
    image_tensor = image_tensor.to('cpu')  # 可视化阶段使用CPU，避免显存问题
    x01 = denorm_if_needed(image_tensor, assume_imagenet_norm)
    score = blue_score_map(x01)  # [B, 1, H, W]
    mask = soft_mask(score, blue_k, blue_t)  # [B, 1, H, W]

    img = x01[0].permute(1, 2, 0).cpu()
    mask_img = mask[0, 0].cpu()

    # 创建一个 1x4 的子图，分别显示原图、蓝色分数热力图、软掩码和 overlay 图
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(score[0, 0].cpu(), cmap='viridis')
    axes[1].set_title("Blue Score")
    axes[1].axis('off')

    axes[2].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title("Soft Mask")
    axes[2].axis('off')

    # 创建 overlay 图
    overlay = img.clone()
    overlay[..., 0] = (overlay[..., 0] * (1 - 0.6 * mask_img) + 0.6 * mask_img).clamp(0, 1)  # 红色高亮 mask
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[3].axis('off')

    # 保存结果
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# 批量处理输入目录中的所有图片
for image_name in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_name)
    if os.path.isfile(image_path):
        print(f"Processing {image_name}...")
        image_tensor = load_image(image_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}_result.png")
        visualize_and_save(image_tensor, output_path)

print("Processing complete.")
