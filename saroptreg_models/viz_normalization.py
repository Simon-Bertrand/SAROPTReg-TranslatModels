from typing import Dict

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


def histogram_equalization(
    img: np.ndarray | torch.Tensor, tileGridSize=(26, 26)
) -> np.ndarray | torch.Tensor:
    if tensor_tag := isinstance(img, torch.Tensor):
        img = img.numpy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGridSize)
    final = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    if tensor_tag:
        return torch.from_numpy(final)
    return final


def _norm_opt(ans: torch.Tensor, type="float", tileGridSize=(26, 26)):
    # ans : CHW tensor in [0,255]
    # out : HWC tensor in [0,1] if float else [0,255]
    he = histogram_equalization(
        ans.to(torch.uint8).moveaxis(0, -1), tileGridSize=tileGridSize
    )
    if type == "float":
        return he / 255
    else:
        return he.to(torch.uint8)  # type: ignore


def _norm_sar(ans: torch.Tensor):
    # ans : CHW tensor in dB
    # out : HWC tensor in [0,1]
    ansLin = 10 ** (ans / 20)
    ansLin = (ansLin / (ansLin.mean() + ansLin.std())).clamp(0, 1).moveaxis(0, -1)
    return ansLin


def normalize_opt_for_viz(sample: Dict[str, torch.Tensor], stats, type="float"):
    return _norm_opt(
        sample["opt"] * stats["std"].reshape(3, 1, 1) + stats["mean"].reshape(3, 1, 1),
        type,
    )


def normalize_sar_for_viz(sample: Dict[str, torch.Tensor], stats):
    ans = _norm_sar(
        sample["template"] * stats["std"].reshape(1, 1, 1)
        + stats["mean"].reshape(1, 1, 1)
    )
    return ans


def plot_opt_with_template_overlay(sample, batch, dataset):
    overlay = normalize_opt_for_viz(dict(opt=sample.im[batch]), dataset.opt_stats)
    template = normalize_sar_for_viz(
        dict(template=sample.templates[batch]), dataset.sar_stats
    ).squeeze(-1)
    fig, axs = plt.subplots(1, 3, figsize=(14, 7))
    th, tw = template.shape
    cy, cx = sample.ijPred[batch].cpu().numpy()  # (y, x) order
    y0 = int(cy - th // 2)
    x0 = int(cx - tw // 2)
    y2 = min(y0 + th, overlay.shape[0])
    x2 = min(x0 + tw, overlay.shape[1])
    ty1 = max(y0, 0) - y0
    tx1 = max(x0, 0) - x0
    ty2 = th - (y0 + th - y2)
    tx2 = tw - (x0 + tw - x2)
    overlay[y0:y2, x0:x2, :] = template[ty1:ty2, tx1:tx2][..., None]
    axs[0].imshow(overlay)
    axs[0].set_title("Opt 512x512 with SAR Template Overlay")
    axs[1].imshow(template, cmap="gray")
    axs[1].set_title("SAR Template 128x128")
    axs[1].axis("off")
    axs[2].imshow(sample.output[batch].detach().squeeze(), cmap="gray")
    axs[2].set_title("Model Output")
    axs[2].axis("off")
    gt_y, gt_x = sample.ijGTruth[batch].cpu().numpy()
    pred_y, pred_x = sample.ijPred[batch].cpu().numpy()
    axs[0].plot(gt_x, gt_y, "go", markersize=13, label="Ground Truth", alpha=0.7)
    axs[0].plot(pred_x, pred_y, "bx", markersize=10, label="Predicted")
    axs[0].legend()
    plt.tight_layout()
    plt.show()
