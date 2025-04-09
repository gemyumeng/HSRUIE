# import torch
# import clip
# from PIL import Image
# import matplotlib.pyplot as plt
#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("RN50", device=device)
#
# # 定义一个列表用于存储特征
# features = []
#
# # 定义钩子函数
# def hook(module, input, output):
#     features.append(output.detach().cpu())
#
# # 注册钩子到模型的特定层，例如最后一个卷积层
# layer = model.visual.relu  # RN50的第四个残差块
# handle = layer.register_forward_hook(hook)
#
# # 加载和预处理图像
# image = preprocess(Image.open("F:/uodd/JPEGImages_raw_test/000014.jpg")).unsqueeze(0).to(device)
# text = clip.tokenize([
#     "The objects in the image are clearly identifiable",
#     "The objects in the image are not clearly identifiable"
# ]).to(device)
#
# with torch.no_grad():
#     # 前向传播，特征将被钩子函数捕获
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)
#
# # 提取存储的特征
# feature_map = features[0]  # 获取钩子函数保存的特征，形状为 (1, C, H, W)
#
# # 将所有通道的特征图合并为一个特征图
# # 方法1：求平均值
# aggregated_feature = feature_map.mean(dim=1).squeeze()
#
# # 方法2：求和（如果需要）
# # aggregated_feature = feature_map.sum(dim=1).squeeze()
#
# # 方法3：计算L2范数
# # aggregated_feature = torch.norm(feature_map, p=2, dim=1).squeeze()
#
# # 将特征图归一化到0-1范围，方便可视化
# # aggregated_feature = aggregated_feature.float()
# # aggregated_feature = (aggregated_feature - aggregated_feature.min()) / (aggregated_feature.max() - aggregated_feature.min())
#
# # 可视化特征图
# plt.imshow(aggregated_feature)
# plt.axis('off')
# plt.show()
#
# # 取消钩子注册（可选）
# handle.remove()


import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载CLIP模型和预处理方法
model, preprocess = clip.load("RN50", device=device)

# 定义一个字典用于存储特征
features = {}


# 定义钩子函数工厂
def get_activation(name):
    def hook(model, input, output):
        features[name] = output.detach().cpu()

    return hook

# 在指定层注册钩子函数
# 初始的ReLU激活后的特征，我们可以注册在bn1后，并在钩子函数中手动应用ReLU
model.visual.relu3.register_forward_hook(get_activation('relu3'))

# 注册后续层的钩子
model.visual.layer1.register_forward_hook(get_activation('layer1'))
model.visual.layer2.register_forward_hook(get_activation('layer2'))
model.visual.layer3.register_forward_hook(get_activation('layer3'))
model.visual.layer4.register_forward_hook(get_activation('layer4'))

# 加载和预处理图像
image = preprocess(Image.open("/home/Data_yuanbao/ym2/CLIP-main/000052.png")).unsqueeze(0).to(device)

# 运行前向传播，钩子函数将捕获特征
with torch.no_grad():
    image_features = model.encode_image(image)


# 对每个层的特征进行处理和可视化
for layer_name in ['relu3', 'layer1', 'layer2', 'layer3', 'layer4']:
    feature_map = features[layer_name]  # 获取该层的特征，形状为 (1, C, H, W)
    feature_map = feature_map.float()
    print(feature_map.size())
    # 如果是bn1层的特征，需要手动应用ReLU激活
    # if layer_name == 'relu1':
    #     feature_map = torch.relu(feature_map)

    # 将特征在通道维度上求平均，得到单通道的特征图
    aggregated_feature = feature_map.mean(dim=1).squeeze()  # 形状为 (H, W)

    # 对特征图进行归一化处理
    aggregated_feature = (aggregated_feature - aggregated_feature.min()) / (
                aggregated_feature.max() - aggregated_feature.min())

    # 可视化特征图
    plt.figure(figsize=(6, 6))
    plt.imshow(aggregated_feature)
    plt.title(f'Feature Map after {layer_name}')
    plt.axis('off')
    plt.show()


