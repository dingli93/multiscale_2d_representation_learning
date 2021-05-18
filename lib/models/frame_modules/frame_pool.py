import torch
from torch import nn

class FrameAvgPool(nn.Module):

    def __init__(self, cfg):
        super(FrameAvgPool, self).__init__()
        input_size = cfg.INPUT_SIZE
        hidden_size = cfg.HIDDEN_SIZE
        kernel_size = cfg.KERNEL_SIZE
        stride = cfg.STRIDE
        self.multi_scale = cfg.multi_scale_2d_map
        # self.scale = cfg.scale
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.avg_pool = nn.AvgPool1d(kernel_size, stride)
        if self.multi_scale:
            # 256 is temporal length of feature in Charades, modify it when in other datasets.
            self.avg_pool_list = [nn.AvgPool1d(kernel_size_num, 256/kernel_size_num) for kernel_size_num in kernel_size]

    def forward(self, visual_input):
        if not self.multi_scale:
            vis_h = torch.relu(self.vis_conv(visual_input))
            vis_h = self.avg_pool(vis_h)
            return vis_h
        else:
            vis_h_list = []
            for i, avg_pool in enumerate(self.avg_pool_list):
                vis_h = torch.relu(self.vis_conv(visual_input))
                vis_h = avg_pool(vis_h)
                vis_h_list.append(vis_h)
            return vis_h_list

class FrameMaxPool(nn.Module):

    def __init__(self, input_size, hidden_size, stride):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(input_size, hidden_size, 1, 1)
        self.max_pool = nn.MaxPool1d(stride)

    def forward(self, visual_input):
        vis_h = torch.relu(self.vis_conv(visual_input))
        vis_h = self.max_pool(vis_h)
        return vis_h
