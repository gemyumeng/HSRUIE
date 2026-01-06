import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
class MIDDLE(nn.Module):
    def __init__(self):
        super(MIDDLE, self).__init__()
        self.low = DetailEnhancement2(img_dim = 128, feature_dim = 128, norm = nn.GroupNorm, act = nn.ReLU)
        self.high = FEM(128,128)

    def forward(self, input, x):
        # 多尺度分支提取
        x1 = self.low(input, x)
        x2 = self.high(x)
        out = x1 * 0.5 + x2 * 0.5
        return out


class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=4):
        super(FEM, self).__init__()
        self.scale = 1
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )

        # 分支0：局部上下文（小感受野）
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )

        # 分支1：中尺度上下文（扩张卷积）
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        # 分支2：全局上下文（大感受野）
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1),  # 全局池化
            nn.Conv2d(inter_planes, 2 * inter_planes, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 动态融合权重
        self.fusion_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(6 * inter_planes, 3, kernel_size=1, bias=False),
            nn.Softmax(dim=1)
        )

        # 特征压缩
        self.ConvLinear = BasicConv(2 * inter_planes, out_planes, kernel_size=1, stride=1,padding=0, relu=True)

        # 残差连接
        self.shortcut = BasicConv(out_planes, out_planes, kernel_size=3, stride=1,padding=1, relu=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        # 多尺度分支提取
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x00 = self.branch(x)
        x2 = self.branch2(x) * x00  # 将全局上下文作为权重调节原始特征

        # 特征融合
        merged = torch.cat((x0, x1, x2), dim=1)

        # 动态权重融合
        weights = self.fusion_weights(merged)  # (B, 3, 1, 1)
        weighted = x0 * weights[:, 0:1] + x1 * weights[:, 1:2] + x2 * weights[:, 2:3]

        # 压缩输出通道
        out = self.ConvLinear(weighted)
        out = self.shortcut(out)

        # 残差连接和激活
        out = out * self.scale + x

        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.GroupNorm(int(out_planes/2), out_planes)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class DetailEnhancement2(nn.Module):
    def __init__(self, img_dim, feature_dim, norm, act):
        super(DetailEnhancement2, self).__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, kernel_size=3,stride=2 ,padding=1, bias=False),
            norm(32, img_dim),
            act()
        )
        self.img_er = MEEM(img_dim, img_dim // 2, 3, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(32, feature_dim),
            act(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(32, feature_dim),
            act(),
        )

        self.feature_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(32, feature_dim),
            act(),

        )

    def forward(self, img, feature):
        feature = self.feature_conv(feature)   # feature_dim

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature   # img_dim

        # out_feature = torch.cat([feature, img_feature], dim=1) # feature_dim + img_dim
        out_feature = feature + img_feature
        out_feature = self.fusion_conv(out_feature)

        return out_feature


class DetailEnhancement(nn.Module):
    def __init__(self, img_dim, feature_dim, norm, act):
        super(DetailEnhancement, self).__init__()
        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, img_dim, 3, padding=1, bias=False),
            norm(32, img_dim),
            act()
        )
        self.img_er = MEEM(img_dim, img_dim // 2, 3, norm, act)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(32, feature_dim),
            act(),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(32, feature_dim),
            act(),
        )

        self.feature_conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1, bias=False),
            norm(32, feature_dim),
            act(),

        )

    def forward(self, img, feature):
        feature = self.feature_conv(feature)   # feature_dim

        img_feature = self.img_in_conv(img)
        img_feature = self.img_er(img_feature) + img_feature   # img_dim

        # out_feature = torch.cat([feature, img_feature], dim=1) # feature_dim + img_dim
        out_feature = feature + img_feature
        out_feature = self.fusion_conv(out_feature)

        return out_feature


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim, norm, act):
        super(EdgeEnhancer, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim, 1, bias=False),
            norm(32, in_dim),
            nn.ReLU()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 1, bias=False),
            norm(32, in_dim),
            nn.ReLU()
        )
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def high_pass_filter(self, x):
        """
        频域高通滤波，用于提取高频信息
        """
        _, _, h, w = x.size()
        # 构建频域网格坐标
        y, x_grid = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device))  # 确保在同一设备
        # 计算中心点位置
        center_y, center_x = h // 2, w // 2
        distance = torch.sqrt((x_grid - center_x) ** 2 + (y - center_y) ** 2)  # 距离计算

        # 构建高通滤波器
        radius = min(h, w) // 4  # 截止频率
        high_pass = torch.sigmoid(10 * (distance - radius)).to(x.device)  # 确保滤波器在相同设备

        # 应用高通滤波器
        freq = fft.fft2(x, dim=(-2, -1))
        freq_shifted = fft.fftshift(freq)
        high_freq = freq_shifted * high_pass[None, None, :, :]  # 确保所有张量在同一设备
        high_freq_spatial = torch.real(fft.ifft2(fft.ifftshift(high_freq), dim=(-2, -1)))

        return high_freq_spatial

    def forward(self, x):
        low_freq = self.pool(x)
        edge = x - low_freq
        high_freq = self.high_pass_filter(x)
        # refined_edge = edge + high_freq
        refined_edge = torch.cat([edge, high_freq], dim=1)
        refined_edge = self.fusion_conv(refined_edge)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(out)
        enhanced_edge = refined_edge * sa
        enhanced_edge = self.out_conv(enhanced_edge)
        return enhanced_edge + x


class MEEM(nn.Module):
    def __init__(self, in_dim, hidden_dim, width, norm, act):
        super(MEEM, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            norm(32, hidden_dim),
            nn.Sigmoid(),
            EdgeEnhancer(hidden_dim, norm, act)
        )

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for i in range(width - 1):
            self.mid_conv.append(nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
                norm(32, hidden_dim),
                nn.Sigmoid()
            ))
            self.edge_enhance.append(EdgeEnhancer(hidden_dim, norm, act))

        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * width, in_dim, 1, bias=False),
            norm(32, in_dim),
            act()
        )

    def forward(self, x):
        mid = self.in_conv(x)

        out = mid
        # print(out.shape)

        for i in range(self.width - 1):
            mid = self.pool(mid)
            mid = self.mid_conv[i](mid)

            out = torch.cat([out, self.edge_enhance[i](mid)], dim=1)

        out = self.out_conv(out)

        return out
