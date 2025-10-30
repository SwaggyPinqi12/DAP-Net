import argparse
import logging
import os
import re
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images or a folder of images')
    parser.add_argument('--model', '-m', default='./checkpoints/Gen/DAP/best_epoch1000.pkl', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--net', type=str, default='NDO', help='Network type')
    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 需要预测的两个目录
    # input_path = args.input
    input_dirs = [
        r'/mnt/e/Run/DefectGen/data/dataset/6648_1k/256/bw_t6v2te2/images/testing',
        r'/mnt/e/Run/DefectGen/data/dataset/6648_1k/256/free'
    ]

    # 获取pkl文件父目录的父目录名和父目录名
    model_path = os.path.abspath(args.model)
    model_parent = os.path.basename(os.path.dirname(os.path.dirname(model_path)))  # 父目录的父目录名
    model_name_wo_ext = os.path.basename(os.path.dirname(model_path))              # 父目录名

    # 提取pkl文件名末尾连续数字
    model_file = os.path.splitext(os.path.basename(model_path))[0]
    match = re.search(r'(\d+)$', model_file)
    last_digits = match.group(1) if match else 'default'

    # 构造新的输出目录
    output_base_dir = os.path.join(os.getcwd(), 'output', model_parent, model_name_wo_ext, last_digits)
    output_mask_dir = os.path.join(output_base_dir, 'mask')
    output_map_dir = os.path.join(output_base_dir, 'map')
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_map_dir, exist_ok=True)

    # 动态导入网络
    if args.net == 'DAP':
        from unet.DAP import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    elif args.net == 'SA':
        from unet.DAP_SA import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    elif args.net == 'CAA':
        from unet.DAP_CAA import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    elif args.net == 'Sim':
        from unet.DAP_SimAM import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    elif args.net == 'BAM':
        from unet.DAP_BAM import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    elif args.net == 'EMA':
        from unet.DAP_EMA import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    elif args.net == 'SegNext':
        from unet.DAP_SegNext import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    elif args.net == 'NDO':
        from unet.DAPnoDenseor import DAPNet
        net = DAPNet(n_channels=1, n_classes=args.classes, m=3)
    else:
        raise ValueError(f"Unknown network type: {args.net}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')
    net.to(device=device)
    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint['model_state_dict']
    net.load_state_dict(state_dict)
    mask_values = [0, 1]
    logging.info('Model loaded!')

    for input_path in input_dirs:
        logging.info(f'Predicting images in directory: {input_path}')
        if os.path.isdir(input_path):
            in_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('.bmp', '.jpg', '.jpeg'))]
        else:
            in_files = [input_path]

        for filename in in_files:
            img = Image.open(filename)
            net.eval()
            img_tensor = torch.from_numpy(BasicDataset.preprocess(None, img, args.scale, is_mask=False))
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                output = net(img_tensor).cpu()
                output_resized = F.interpolate(output, (img.size[1], img.size[0]), mode='bilinear')

                # 保存概率图/得分图
                if args.classes > 1:
                    prob_map = torch.softmax(output_resized, dim=1)[0, 1].numpy()  # shape: [classes, H, W]
                else:
                    prob_map = torch.sigmoid(output_resized)[0, 0].numpy()

                map_filename = os.path.splitext(os.path.basename(filename))[0] + '.npy'
                np.save(os.path.join(output_map_dir, map_filename), prob_map)

                # 掩膜处理（如需保存mask，取消注释）
                if args.classes > 1:
                    mask = output_resized.argmax(dim=1)
                else:
                    mask = torch.sigmoid(output_resized) > args.mask_threshold
                mask_np = mask[0].long().squeeze().numpy()
                mask_img = mask_to_image(mask_np, mask_values)
                mask_filename = os.path.basename(filename)
                mask_img.save(os.path.join(output_mask_dir, mask_filename))

            if args.viz:
                logging.info(f'Visualizing results for image {filename}, close to continue...')
                plot_img_and_mask(img, mask_np)