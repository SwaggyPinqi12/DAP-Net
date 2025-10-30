import os
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve, auc
from skimage import measure
from tqdm import tqdm
from calcutils import ader_evaluator

def evaluation_from_npy(score_dir, mask_defect_dir, mask_normal_dir):
    """
    生成与utils.py中evaluation_batch一致的pr_list_px, pr_list_sp, gt_list_px, gt_list_sp
    score_dir: 推理得分图npy目录
    mask_defect_dir: 有缺陷mask目录
    mask_normal_dir: 无缺陷mask目录
    """
    pr_list_px = []
    pr_list_sp = []
    gt_list_px = []
    gt_list_sp = []

    # 处理有缺陷图像
    for fname in tqdm(os.listdir(mask_defect_dir), desc="Defect"):
        base_name = os.path.splitext(fname)[0]
        score_path = os.path.join(score_dir, base_name + ".npy")
        mask_path = os.path.join(mask_defect_dir, fname)
        if not os.path.exists(score_path):
            continue
        score = np.load(score_path)
        mask = cv2.imread(mask_path, 0)
        mask_bin = (mask > 127).astype(np.uint8)
        pr_list_px.append(score)
        gt_list_px.append(mask_bin)
        pr_list_sp.append(score.max())
        gt_list_sp.append(1)  # 有缺陷

    # 处理无缺陷图像
    for fname in tqdm(os.listdir(mask_normal_dir), desc="Normal"):
        base_name = os.path.splitext(fname)[0]
        score_path = os.path.join(score_dir, base_name + ".npy")
        if not os.path.exists(score_path):
            continue
        score = np.load(score_path)
        mask = np.zeros_like(score, dtype=np.uint8)
        pr_list_px.append(score)
        gt_list_px.append(mask)
        pr_list_sp.append(score.max())
        gt_list_sp.append(0)  # 无缺陷

    pr_list_px = np.stack(pr_list_px)
    gt_list_px = np.stack(gt_list_px)
    pr_list_sp = np.array(pr_list_sp)
    gt_list_sp = np.array(gt_list_sp)
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = ader_evaluator(pr_list_px, pr_list_sp, gt_list_px, gt_list_sp)
    return auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px

# 示例调用
if __name__ == "__main__":
    
    score_dir = r"/mnt/e/Run/DefectGen/Anomaly/DAP-Net/output/Gen/DAP/1000/map"
    mask_defect_dir = r"/mnt/e/Run/DefectGen/data/dataset/6648_1k/256/bw_t6v2te2/annotations/testing"
    mask_normal_dir = r"/mnt/e/Run/DefectGen/data/dataset/6648_1k/256/free/masks"
    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = evaluation_from_npy(
        score_dir, mask_defect_dir, mask_normal_dir
    )
    result_str = (
        f"I-AUROC: {auroc_sp:.4f}\n"
        f"I-AP: {ap_sp:.4f}\n"
        f"I-F1: {f1_sp:.4f}\n"
        f"P-AUROC: {auroc_px:.4f}\n"
        f"P-AP: {ap_px:.4f}\n"
        f"P-F1: {f1_px:.4f}\n"
        f"P-AUPRO: {aupro_px:.4f}\n"
    )
    print(result_str)

    id_dir = os.path.dirname(score_dir.rstrip("/\\"))
    id_name = os.path.basename(id_dir)
    out_txt = os.path.join(os.path.dirname(id_dir), f"{id_name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(result_str)