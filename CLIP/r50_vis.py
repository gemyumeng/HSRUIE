import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载预训练的ResNet-50模型
model = models.resnet50(pretrained=True).to(device)
model.eval()  # 设置为评估模式

# 定义一个字典用于存储特征
features = {}


# 定义钩子函数工厂
def get_activation(name):
    def hook(model, input, output):
        features[name] = output.detach().cpu()

    return hook


# 在指定层注册钩子函数
# 初始的ReLU激活后的特征（在maxpool层之后）
model.relu.register_forward_hook(get_activation('relu'))
model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))
model.layer3.register_forward_hook(get_activation('layer3'))
model.layer4.register_forward_hook(get_activation('layer4'))

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet的均值
        std=[0.229, 0.224, 0.225]  # ImageNet的标准差
    ),
])

# 加载和预处理图像
image_path = "F:/uodd/JPEGImages_tacl_test/000014.bmp"  # 替换为您的图像路径
original_image = Image.open(image_path).convert('RGB')
image = preprocess(original_image).unsqueeze(0).to(device)

# 运行前向传播，钩子函数将捕获特征
with torch.no_grad():
    output = model(image)

# 对每个层的特征进行处理和可视化
for layer_name in ['relu', 'layer1', 'layer2', 'layer3', 'layer4']:
    feature_map = features[layer_name][0]  # 获取该层的特征，形状为 (C, H, W)
    feature_map = feature_map.float()
    # 将特征在通道维度上求平均，得到单通道的特征图
    aggregated_feature = feature_map.mean(dim=0).numpy()  # 形状为 (H, W)

    # 将特征图归一化到0-1
    aggregated_feature -= aggregated_feature.min()
    aggregated_feature /= aggregated_feature.max()

    # 可视化特征图
    plt.figure(figsize=(8, 8))
    plt.imshow(aggregated_feature, cmap='viridis')
    plt.title(f'Feature Map after {layer_name}')
    plt.axis('off')
    plt.show()
