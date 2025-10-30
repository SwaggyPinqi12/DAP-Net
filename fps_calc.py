import time
import torch
import numpy as np


from unet.DAP import DAPNet
net = DAPNet(n_channels=1, n_classes=2, m=2)

net.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)

# x是输入图片的大小
x = torch.zeros((1,1,256,256)).cuda()
t_all = []

for i in range(1000):
    t1 = time.time()
    y = net(x)
    t2 = time.time()
    t_all.append(t2 - t1)

print('average time:', np.mean(t_all) / 1)
print('average fps:',1 / np.mean(t_all))

print('fastest time:', min(t_all) / 1)
print('fastest fps:',1 / min(t_all))

print('slowest time:', max(t_all) / 1)
print('slowest fps:',1 / max(t_all))