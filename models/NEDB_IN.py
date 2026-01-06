import torch
from torch import nn


# 4个 conv 的 Dense block
class Dense_Block_IN(nn.Module):
    def __init__(self, block_num, inter_channel, channel):
        super(Dense_Block_IN, self).__init__()
        #
        concat_channels = channel + block_num * inter_channel
        channels_now = channel

        self.group_list = nn.ModuleList([])
        for i in range(block_num):
            group = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channel, kernel_size=3,
                          stride=1, padding=1),
                nn.ReLU(),
            )
            self.add_module(name='group_%d' % i, module=group)
            self.group_list.append(group)

            channels_now += inter_channel

        assert channels_now == concat_channels
        #
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        #

    def forward(self, x):
        feature_list = [x]

        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)

        inputs = torch.cat(feature_list, dim=1)
        #
        fusion_outputs = self.fusion(inputs)
        #
        block_outputs = fusion_outputs + x

        return block_outputs

class Dense_Block_IN_Tanh(nn.Module):
    def __init__(self, block_num, inter_channel, channel):
        super(Dense_Block_IN_Tanh, self).__init__()
        #
        concat_channels = channel + block_num * inter_channel
        channels_now = channel

        self.group_list = nn.ModuleList([])
        for i in range(block_num):
            group = nn.Sequential(
                nn.Conv2d(in_channels=channels_now, out_channels=inter_channel, kernel_size=3,
                          stride=1, padding=1),
                nn.Tanh(),
            )
            self.add_module(name='group_%d' % i, module=group)
            self.group_list.append(group)

            channels_now += inter_channel

        assert channels_now == concat_channels
        #
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, channel, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )
        #

    def forward(self, x):
        feature_list = [x]

        for group in self.group_list:
            inputs = torch.cat(feature_list, dim=1)
            outputs = group(inputs)
            feature_list.append(outputs)

        inputs = torch.cat(feature_list, dim=1)
        #
        fusion_outputs = self.fusion(inputs)
        #
        block_outputs = fusion_outputs + x

        return block_outputs


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"
    device_ids = [0, 1]
    net = Dense_Block_IN(block_num=3, inter_channel=32, channel=3)
    if torch.cuda.is_available():
        # net_A = nn.DataParallel(net_A, device_ids=device_ids)
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net.cuda()
    # net = Dense_Block_IN(block_num=3, inter_channel=32, channel=3).cuda()
    input1 = torch.FloatTensor(2, 3, 256, 256).cuda()
    t = net(input1)
    print(t)