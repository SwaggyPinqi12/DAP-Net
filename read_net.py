import torch
import os

from unet.DAP import DAPNet

# 2. 实例化网络（根据你的实际参数填写）
# model = UNet(n_channels=1, n_classes=2, bilinear=False)
model = DAPNet(n_channels=1, n_classes=2, m=3)

# 3. 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total model parameters: {total_params:,}")

# 4. 计算FLOPs（需安装thop库）
try:
    from thop import profile
    dummy_input = torch.randn(1, 1, 256, 256)  # 根据你的输入尺寸调整
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"FLOPs (multiply-adds): {flops:,}")
except ImportError:
    print("thop 未安装，无法自动计算FLOPs。可用 pip install thop 安装。")
except Exception as e:
    print(f"FLOPs 计算失败: {e}")
