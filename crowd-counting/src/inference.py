# inference_v3.py
"""
パラメータ調整対応版の推論モジュール
- scaling_coeff: 正規化係数（過検出時は上げる）
- bg_thresh_multiplier: 背景閾値倍率（背景ノイズカット強度）
"""
import torch
import torch.nn as nn
import torchvision.ops as ops
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from io import BytesIO

from src.utils import (
    draw_bounding_boxes, convert_4corners_to_x1y1x2y2,
    get_features, bboxes_tointeger, compute_avg_conv_filter,
    rescale_tensor, resize_conv_maps, rescale_bbox, ellipse_coverage
)
from src.model import VisualBackbone

# ------------------- モデルと設定のグローバルロード -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = 'dinov2_vitl14_reg'
resize_dim = 840

model = VisualBackbone(model_name, img_size=resize_dim).eval().to(device)

transform = T.Compose([
    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

map_keys = ['vit_out']


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def get_default_config(scaling_coeff=1.0, bg_thresh_multiplier=1.0):
    """調整可能なパラメータを含むConfig生成"""
    return Config(
        divide_et_impera=True,
        divide_et_impera_twice=False,
        normalize_features=False,
        filter_background=True,
        ellipse_normalization=True,
        ellipse_kernel_cleaning=False,
        exemplar_avg=False,
        correct_bbox_resize=True,
        use_roi_norm=True,
        roi_norm_after_mean=True,
        use_threshold=False,
        cosine_similarity=False,
        use_minmax_norm=True,
        remove_bbox_intersection=False,
        scaling_coeff=scaling_coeff,
        bg_thresh_multiplier=bg_thresh_multiplier,
        fixed_norm_coeff=None,
        num_exemplars=None
    )


# ------------------- 後処理関数 -------------------
def post_process_density_map(conv_maps, pooled_feats, bboxes, output_sizes, config):
    """密度マップの後処理（パラメータ調整対応）"""
    if config.use_threshold:
        output, resize_ratios = resize_conv_maps(conv_maps)
        output = output.mean(dim=0)
        if config.use_minmax_norm:
            output = rescale_tensor(output)
        thresh = torch.median(output)
        output[output < thresh] = 0
        return output

    if config.use_roi_norm and config.roi_norm_after_mean:
        output, resize_ratios = resize_conv_maps(conv_maps)
        output = output.mean(dim=0)
        if config.use_minmax_norm:
            output = rescale_tensor(output)
        pooled_vals = []
        for bbox, ratio in zip(bboxes, resize_ratios):
            scaled_bbox = torch.tensor([
                bbox[0] * ratio[1], bbox[1] * ratio[0],
                bbox[2] * ratio[1], bbox[3] * ratio[0]
            ]).int()
            output_size = (
                int(scaled_bbox[3] - scaled_bbox[1]),
                int(scaled_bbox[2] - scaled_bbox[0])
            )
            pooled = ops.roi_align(
                output.unsqueeze(0).unsqueeze(0),
                [scaled_bbox.unsqueeze(0).float().to(device)],
                output_size=output_size, spatial_scale=1.0
            )
            pooled_vals.append(pooled)

        # 正規化係数の計算（scaling_coeffで調整可能）
        if config.ellipse_normalization:
            norm_coeff = sum([
                (p[0, 0] * ellipse_coverage(p.shape[-2], p.shape[-1]).to(device)).sum()
                for p in pooled_vals
            ]) / (len(pooled_vals) * config.scaling_coeff)
        else:
            norm_coeff = sum([p.sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)

        if config.fixed_norm_coeff is not None:
            norm_coeff = config.fixed_norm_coeff

        output = output / norm_coeff

        # 背景フィルタリング（bg_thresh_multiplierで調整可能）
        if config.filter_background:
            thresh = max([f.shape[-2] * f.shape[-1] for f in pooled_feats])
            thresh = (1 / thresh) * config.bg_thresh_multiplier  # 倍率適用
            output[output < thresh] = 0

    return output


# ------------------- 可視化関数 -------------------
def create_visualizations(img_pil: Image.Image, density_map: torch.Tensor, exemplars: list, pred_count: float):
    """
    論文Figure 5風に調整:
    - colormap: 'jet' (青低密度 → 赤高密度)
    - オーバーレイ: alpha=0.5でオリジナルに重ね、密度強調
    """
    density_np = density_map.cpu().numpy()
    norm = Normalize(vmin=density_np.min(), vmax=density_np.max())
    cmap = cm.get_cmap('jet')

    # 1. プレビュー (オリジナル + 赤枠)
    preview_img = draw_bounding_boxes(img_pil.copy(), exemplars, color="red", width=5, text_background=True)

    # 2. 密度マップ単体 (jet colormap)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(density_np, cmap=cmap, norm=norm)
    ax.set_title(f"Density Map (jet colormap)\nPredicted Count: {pred_count:.1f}", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    density_only = Image.open(buf).convert('RGB')
    plt.close(fig)
    buf.close()

    # 3. オーバーレイ
    density_resized = Image.fromarray(
        (cmap(norm(density_np)) * 255).astype(np.uint8)
    ).resize(img_pil.size, Image.BICUBIC)
    overlay_pil = Image.blend(img_pil.copy(), density_resized.convert('RGB'), alpha=0.5)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay_pil)
    ax.set_title(f"Overlay (Original + Density Map)\nPredicted Count: {pred_count:.1f}", fontsize=14)
    ax.axis('off')
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    overlay_img = Image.open(buf).convert('RGB')
    plt.close(fig)
    buf.close()

    # 4. グリッド表示
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    axs[0].imshow(preview_img)
    axs[0].set_title("Original Image + Exemplars", fontsize=14)
    axs[0].axis('off')

    axs[1].imshow(density_np, cmap=cmap, norm=norm)
    axs[1].set_title("Generated Density Map", fontsize=14)
    fig.colorbar(axs[1].images[0], ax=axs[1], shrink=0.8)
    axs[1].axis('off')

    axs[2].imshow(overlay_pil)
    axs[2].set_title("Overlay on Original", fontsize=14)
    axs[2].axis('off')

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    grid_combined = Image.open(buf).convert('RGB')
    plt.close(fig)
    buf.close()

    return preview_img, density_only, overlay_img, grid_combined, float(pred_count)


# ------------------- メイン推論関数 -------------------
def run_inference_with_params(
    img: Image.Image,
    exemplars: list,
    scaling_coeff: float = 1.0,
    bg_thresh_multiplier: float = 1.0
):
    """
    パラメータ調整対応の推論関数

    Args:
        img: 入力画像
        exemplars: Exemplarのリスト [[x1,y1,x2,y2], ...]
        scaling_coeff: 正規化係数（過検出時は上げる、過少検出時は下げる）
        bg_thresh_multiplier: 背景閾値倍率（背景ノイズカット強度）
    """
    config = get_default_config(
        scaling_coeff=scaling_coeff,
        bg_thresh_multiplier=bg_thresh_multiplier
    )

    img_rgb = img.convert('RGB')
    w, h = img_rgb.size

    # 特徴量抽出
    with torch.no_grad():
        feats = get_features(
            model, img_rgb, transform, map_keys,
            divide_et_impera=config.divide_et_impera,
            divide_et_impera_twice=config.divide_et_impera_twice
        )
        if config.cosine_similarity or config.normalize_features:
            feats = feats / feats.norm(dim=1, keepdim=True)

    # Exemplar処理
    ex_bboxes = [
        convert_4corners_to_x1y1x2y2(b) if len(b) == 4 and isinstance(b[0], list) else b
        for b in exemplars
    ]
    if config.num_exemplars:
        ex_bboxes = ex_bboxes[:config.num_exemplars]

    bboxes = np.array([
        (x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in ex_bboxes
    ]) * feats.shape[-1]
    bboxes = bboxes_tointeger(bboxes, config.remove_bbox_intersection)

    conv_maps = []
    pooled_features_list = []
    rescaled_bboxes = []
    output_sizes = []

    for bbox in bboxes:
        bbox_tensor = torch.tensor(bbox).to(device)
        output_size = (int(bbox_tensor[3] - bbox_tensor[1]), int(bbox_tensor[2] - bbox_tensor[0]))
        pooled = ops.roi_align(
            feats, [bbox_tensor.unsqueeze(0).float()],
            output_size=output_size, spatial_scale=1.0
        )
        if config.ellipse_kernel_cleaning:
            ellipse = ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).unsqueeze(0).unsqueeze(0).to(device)
            pooled *= ellipse
        pooled_features_list.append(pooled)

        if config.exemplar_avg:
            continue

        conv_weights = pooled.view(feats.shape[1], 1, *output_size)
        conv_layer = nn.Conv2d(
            in_channels=feats.shape[1],
            out_channels=1 if config.cosine_similarity else feats.shape[1],
            kernel_size=output_size,
            padding=0,
            groups=1 if config.cosine_similarity else feats.shape[1],
            bias=False
        ).to(device)
        conv_layer.weight = nn.Parameter(pooled if config.cosine_similarity else conv_weights)

        with torch.no_grad():
            output = conv_layer(feats[0])

        rescaled_bbox = rescale_bbox(bbox_tensor, output, feats) if config.correct_bbox_resize else bbox_tensor
        rescaled_bboxes.append(rescaled_bbox)
        conv_maps.append(output)
        output_sizes.append(output_size)

    if config.exemplar_avg:
        pooled = compute_avg_conv_filter(pooled_features_list)
        output_size = pooled.shape[1:]
        conv_weights = pooled.view(pooled.shape[0], 1, *output_size)
        conv_layer = nn.Conv2d(
            in_channels=feats.shape[1],
            out_channels=1 if config.cosine_similarity else feats.shape[1],
            kernel_size=output_size,
            padding=0,
            groups=1 if config.cosine_similarity else feats.shape[1],
            bias=False
        ).to(device)
        conv_layer.weight = nn.Parameter(pooled.unsqueeze(0) if config.cosine_similarity else conv_weights)
        with torch.no_grad():
            output = conv_layer(feats[0])
        conv_maps.append(output)
        output_sizes.append(output_size)

    pred_map = post_process_density_map(
        conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config
    )
    pred_count = pred_map.sum().item()

    # 可視化
    preview, density_only, overlay, grid, count = create_visualizations(
        img_rgb, pred_map, ex_bboxes, pred_count
    )
    return preview, density_only, overlay, grid, count
