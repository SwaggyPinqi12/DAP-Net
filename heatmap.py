# predict_with_heatmaps.py
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
import matplotlib.pyplot as plt
import cv2

from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask

# ---------------------------
# Helpers: heatmap & overlay
# ---------------------------
def normalize_map(x):
    """Normalize numpy array to [0,1]"""
    xm = x - x.min()
    if xm.max() == 0:
        return xm
    return xm / (xm.max() + 1e-8)

def save_heatmap_gray(heat_np, save_path, cmap='jet'):
    """Save a single-channel heatmap as colored image using matplotlib colormap."""
    heat_norm = normalize_map(heat_np)
    # Use OpenCV colormap for speed; convert to 0-255 uint8 first
    heat_uint8 = np.uint8(255 * heat_norm)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # BGR
    # Convert BGR->RGB for saving by cv2 or PIL consistency
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    Image.fromarray(heat_color).save(save_path)

def overlay_on_image(input_pil, heat_np, out_path, alpha=0.5):
    """Overlay heatmap onto input PIL image and save."""
    # ensure heatmap same size as input image
    im = input_pil.convert('RGB')
    W, H = im.size
    heat_resized = cv2.resize(np.float32(heat_np), (W, H), interpolation=cv2.INTER_LINEAR)
    heat_norm = normalize_map(heat_resized)
    heat_uint8 = np.uint8(255 * heat_norm)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)  # BGR
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    img_np = np.array(im, dtype=np.uint8)

    overlay = cv2.addWeighted(img_np, 1 - alpha, heat_color, alpha, 0)
    Image.fromarray(overlay).save(out_path)

# ---------------------------
# Argument parsing (your original)
# ---------------------------
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images or a folder of images (with heatmap saving)')
    parser.add_argument('--model', '-m', default='./checkpoints/Gen/DAP/best_epoch1000.pkl', metavar='FILE',metavar='FILE',
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
    parser.add_argument('--net', type=str, default='DAP', choices=['DAP', 'NDO'], help='Network type')
    return parser.parse_args()

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # input dirs (你原来的两个目录)
    input_dirs = [
        r'/mnt/e/Run/DefectGen/Anomaly/DAP-Net/data/illustration/imgs',
    ]

    # 构造输出目录（沿用你原脚本逻辑）
    model_path = os.path.abspath(args.model)
    model_parent = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    model_name_wo_ext = os.path.basename(os.path.dirname(model_path))
    model_file = os.path.splitext(os.path.basename(model_path))[0]
    match = re.search(r'(\d+)$', model_file)
    last_digits = match.group(1) if match else 'default'

    output_base_dir = os.path.join(os.getcwd(), 'heatmap', model_parent, model_name_wo_ext, last_digits)
    output_mask_dir = os.path.join(output_base_dir, 'mask')
    output_map_dir = os.path.join(output_base_dir, 'map')
    output_feature_dir = os.path.join(output_base_dir, 'feature_maps')
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(output_map_dir, exist_ok=True)
    os.makedirs(output_feature_dir, exist_ok=True)

    # 动态导入网络（与你原脚本保持一致）
    if args.net == 'DAP':
        from unet.DAP import DAPNet
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

    # ---------------------------
    # Register forward hooks to capture intermediate features
    # ---------------------------
    target_layer_names = [
        'conv1_0', 'conv2_0', 'conv3_0', 'conv4_0', 'conv5_0',
        'adapter', 'upbranch', 'upcat'
    ]

    feature_maps = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            """安全地保存层输出，无论是 tensor 还是 list/tuple"""
            try:
                if isinstance(out, torch.Tensor):
                    feature_maps[name] = out.detach().cpu()
                elif isinstance(out, (list, tuple)):
                    # 只保存其中是 tensor 的部分
                    out_tensors = []
                    for i, o in enumerate(out):
                        if isinstance(o, torch.Tensor):
                            out_tensors.append(o.detach().cpu())
                    if len(out_tensors) > 0:
                        # 如果多个输出，堆叠在第一维方便后续可视化
                        feature_maps[name] = out_tensors
                else:
                    logging.warning(f"Hook at {name}: unsupported output type {type(out)}")
            except Exception as e:
                logging.exception(f"Hook error at {name}: {e}")
        return hook

    # 注册到模型上
    for name, module in net.named_modules():
        if name in target_layer_names:
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
            logging.info(f"Registered hook for: {name}")

    # ---------------------------
    # Process images
    # ---------------------------
    for input_path in input_dirs:
        logging.info(f'Predicting images in directory: {input_path}')
        if os.path.isdir(input_path):
            in_files = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.lower().endswith(('.bmp', '.jpg', '.jpeg', '.png'))]
        else:
            in_files = [input_path]

        for filename in in_files:
            logging.info(f'Processing {filename}')
            img = Image.open(filename).convert('L')  # 灰度
            net.eval()
            img_tensor = torch.from_numpy(BasicDataset.preprocess(None, img, args.scale, is_mask=False))
            img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)

            # 清空 feature_maps（避免跨张图混淆）
            feature_maps.clear()

            with torch.no_grad():
                output = net(img_tensor).cpu()
                output_resized = F.interpolate(output, (img.size[1], img.size[0]), mode='bilinear')

                # 保存概率图/得分图
                if args.classes > 1:
                    prob_map = torch.softmax(output_resized, dim=1)[0, 1].numpy()
                else:
                    prob_map = torch.sigmoid(output_resized)[0, 0].numpy()

                # # 保存 map (.npy)
                # map_filename = os.path.splitext(os.path.basename(filename))[0] + '.npy'
                # np.save(os.path.join(output_map_dir, map_filename), prob_map)
                
                # ---- 保存热力图 (heatmap) ----
                if not args.no_save:
                    heatmap = (prob_map * 255).astype(np.uint8)
                    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    heatmap_filename = os.path.splitext(os.path.basename(filename))[0] + '_heatmap.png'
                    cv2.imwrite(os.path.join(output_map_dir, heatmap_filename), heatmap_color)

                # 掩膜处理（如需保存mask）
                if not args.no_save:
                    if args.classes > 1:
                        mask = output_resized.argmax(dim=1)
                    else:
                        mask = torch.sigmoid(output_resized) > args.mask_threshold
                    mask_np = mask[0].long().squeeze().numpy()
                    # 复用你原来的 mask_to_image 转换逻辑（简化版）
                    def mask_to_image(mask_np, mask_values):
                        if isinstance(mask_values[0], list):
                            out = np.zeros((mask_np.shape[-2], mask_np.shape[-1], len(mask_values[0])), dtype=np.uint8)
                        elif mask_values == [0, 1]:
                            out = np.zeros((mask_np.shape[-2], mask_np.shape[-1]), dtype=bool)
                        else:
                            out = np.zeros((mask_np.shape[-2], mask_np.shape[-1]), dtype=np.uint8)
                        if mask_np.ndim == 3:
                            mask_np = np.argmax(mask_np, axis=0)
                        for i, v in enumerate(mask_values):
                            out[mask_np == i] = v
                        return Image.fromarray(out)
                    mask_img = mask_to_image(mask_np, mask_values)
                    mask_filename = os.path.basename(filename)
                    mask_img.save(os.path.join(output_mask_dir, mask_filename))

            # ---------------------------
            # 保存中间特征热力图
            # ---------------------------
            # feature_maps 现在包含 hook 捕获的层（如果模型确实在前向过程触发了这些模块）
            img_basename = os.path.splitext(os.path.basename(filename))[0]
            for layer_name, fmap in feature_maps.items():
                try:
                    # fmap: tensor [B, C, H, W] 或者某些模块可能输出 list/tuple —— 先做保护性处理
                    if isinstance(fmap, (list, tuple)):
                        # 记录第一个输出项
                        fmap_t = fmap[0]
                    else:
                        fmap_t = fmap
                    if fmap_t.ndim == 4:
                        # channel fuse: mean across channels
                        fused = torch.mean(fmap_t, dim=1, keepdim=True)  # [B,1,H,W]
                        # upsample to original image size
                        fused_up = F.interpolate(fused, size=(img.size[1], img.size[0]), mode='bilinear', align_corners=False)
                        heat_np = fused_up[0,0].numpy()
                    elif fmap_t.ndim == 3:
                        # [B, H, W]  or [C, H, W] fallback
                        if fmap_t.shape[0] == 1:
                            heat_np = fmap_t[0].numpy()
                        else:
                            heat_np = np.mean(fmap_t.numpy(), axis=0)
                    else:
                        logging.warning(f"Unsupported fmap shape for layer {layer_name}: {fmap_t.shape}")
                        continue

                    # 保存原始数值 numpy（便于后续分析）
                    npy_path = os.path.join(output_feature_dir, f"{img_basename}_{layer_name}.npy")
                    np.save(npy_path, heat_np)

                    # 保存灰度热力图的伪彩色图
                    color_path = os.path.join(output_feature_dir, f"{img_basename}_{layer_name}_color.png")
                    save_heatmap_gray(heat_np, color_path)

                    # # 保存覆盖在原图上的叠加图
                    # overlay_path = os.path.join(output_feature_dir, f"{img_basename}_{layer_name}_overlay.png")
                    # overlay_on_image(img, heat_np, overlay_path, alpha=0.5)

                    logging.info(f"Saved feature maps for {layer_name}: .npy, color, overlay")
                except Exception as e:
                    logging.exception(f"Failed to process/save feature for layer {layer_name}: {e}")

            # # optional visualization
            # if args.viz:
            #     logging.info(f'Visualizing results for image {filename}, close to continue...')
            #     plot_img_and_mask(img, mask_np)

    # ---------------------------
    # Clean up hooks
    # ---------------------------
    for h in hooks:
        h.remove()
    logging.info("All done. Feature maps saved under: %s", output_feature_dir)
