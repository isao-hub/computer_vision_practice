# inference.py
import os
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

config = Config(
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
    scaling_coeff=1.0,
    fixed_norm_coeff=None,
    num_exemplars=None
)

# ------------------- ユーティリティ関数 -------------------
def combine_pil_and_plot(original_img: Image.Image, density_map: np.ndarray, title: str = "") -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(density_map, cmap='jet')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.axis('off')

    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    plot_img = Image.open(buf)
    plt.close(fig)

    original_img = original_img.convert('RGB')
    plot_img = plot_img.resize(original_img.size)

    combined = Image.new('RGB', (original_img.width + plot_img.width, original_img.height), (255, 255, 255))
    combined.paste(original_img, (0, 0))
    combined.paste(plot_img, (original_img.width, 0))
    buf.close()
    return combined

def post_process_density_map(conv_maps, pooled_feats, bboxes, output_sizes, config):
    # （元のコードと同じ内容を貼り付け）
    # 省略せずそのまま使用してください（変更なし）
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
        if config.ellipse_normalization:
            norm_coeff = sum([(p[0, 0] * ellipse_coverage(p.shape[-2], p.shape[-1]).to(device)).sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
        else:
            norm_coeff = sum([p.sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
        if config.fixed_norm_coeff is not None:
            norm_coeff = config.fixed_norm_coeff
        output = output / norm_coeff
        if config.filter_background:
            thresh = max([f.shape[-2] * f.shape[-1] for f in pooled_feats])
            thresh = (1 / thresh) * 1.0
            output[output < thresh] = 0
    return output

## [カスタマイズ]
# inference.py

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def create_visualizations(img_pil: Image.Image, density_map: torch.Tensor, exemplars: list, pred_count: float):
    """
    論文Figure 5風に調整:
    - colormap: 'jet' (青低密度 → 赤高密度)
    - オーバーレイ: alpha=0.5でオリジナルに重ね、密度強調
    - 出力: preview, density_only, overlay, grid_combined (論文風グリッド)
    """
    density_np = density_map.cpu().numpy()
    norm = Normalize(vmin=density_np.min(), vmax=density_np.max())
    cmap = cm.get_cmap('jet')  # 論文風: 青-緑-黄-赤グラデーション

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

    # 3. オーバーレイ (オリジナルに密度を半透明重ね、論文風強調)
    # サイズをオリジナルにリサイズ
    density_resized = Image.fromarray((cmap(norm(density_np)) * 255).astype(np.uint8)).resize(img_pil.size, Image.BICUBIC)
    overlay_pil = img_pil.copy()
    overlay_pil = Image.blend(overlay_pil, density_resized.convert('RGB'), alpha=0.5)  # Blendで論文風の重ね合わせ (additive風)

    # matplotlibでタイトル追加
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

    # 4. オリジナル画像 / 密度マップ / オーバーレイ を横並び)
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

# def create_visualizations(img_pil: Image.Image, density_map: np.ndarray, exemplars: list, pred_count: float):
#     """
#     3つの画像を生成してタプルで返す
#     1. preview_img: 元画像 + 赤枠
#     2. density_only: 密度マップ単体 (viridis)
#     3. overlay_img: 元画像に密度マップを半透明で重ね合わせ
#     """
#     # 1. 元画像 + 赤枠
#     preview_img = draw_bounding_boxes(img_pil.copy(), exemplars, color="red", width=5, text_background=True)

#     density_np = density_map.cpu().numpy()

#     # viridis colormap で密度マップを描画
#     fig, ax = plt.subplots(figsize=(8, 8))
#     norm = Normalize(vmin=density_np.min(), vmax=density_np.max())
#     im = ax.imshow(density_np, cmap='viridis', norm=norm)
#     ax.set_title(f"Density Map (viridis)\nPredicted Count: {pred_count:.1f}", fontsize=14)
#     fig.colorbar(im, ax=ax, shrink=0.8)
#     ax.axis('off')

#     buf = BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
#     buf.seek(0)
#     density_only = Image.open(buf).convert('RGB')
#     plt.close(fig)
#     buf.close()

#     # 3. オーバーレイ画像（元画像 + 半透明密度マップ）
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(img_pil)
#     im = ax.imshow(density_np, cmap='viridis', alpha=0.6, norm=norm)  # alphaで半透明
#     ax.set_title(f"Overlay on Original Image\nPredicted Count: {pred_count:.1f}", fontsize=14)
#     ax.axis('off')
#     fig.colorbar(im, ax=ax, shrink=0.8)

#     buf = BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
#     buf.seek(0)
#     overlay_img = Image.open(buf).convert('RGB')
#     plt.close(fig)
#     buf.close()

#     return preview_img, density_only, overlay_img

# ------------------- メイン推論関数 -------------------
def run_inference(
    img: Image.Image,
    exemplars: list  # List of [[x1,y1,x2,y2], ...]  (すでにx1y1x2y2形式)
):
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
    ex_bboxes = [convert_4corners_to_x1y1x2y2(b) if len(b) == 4 and isinstance(b[0], list) else b for b in exemplars]
    if config.num_exemplars:
        ex_bboxes = ex_bboxes[:config.num_exemplars]

    bboxes = np.array([(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in ex_bboxes]) * feats.shape[-1]
    bboxes = bboxes_tointeger(bboxes, config.remove_bbox_intersection)

    conv_maps = []
    pooled_features_list = []
    rescaled_bboxes = []
    output_sizes = []

    for bbox in bboxes:
        bbox_tensor = torch.tensor(bbox).to(device)
        output_size = (int(bbox_tensor[3] - bbox_tensor[1]), int(bbox_tensor[2] - bbox_tensor[0]))
        pooled = ops.roi_align(feats, [bbox_tensor.unsqueeze(0).float()], output_size=output_size, spatial_scale=1.0)
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

    pred_map = post_process_density_map(conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config)
    pred_count = pred_map.sum().item()

    # [カスタマイズ]可視化
    preview, density_only, overlay, grid, count = create_visualizations(img_rgb, pred_map, ex_bboxes, pred_count)
    return preview, density_only, overlay, grid, count
    # preview, density_only, overlay = create_visualizations(img_rgb, pred_map, ex_bboxes, pred_count)

    # return preview, density_only, overlay, float(pred_count)

    # old_可視化
    # img_with_boxes = draw_bounding_boxes(img_rgb.copy(), ex_bboxes, color="red", width=5)
    # density_np = pred_map.cpu().numpy()
    # result_img = combine_pil_and_plot(img_with_boxes, density_np, title=f"Predicted Count: {pred_count:.1f}")

    # return result_img, float(pred_count)


# import os, json
# import numpy as np

# import sys
# if '..' not in sys.path:
#     sys.path.append('..')

# from PIL import Image
# from io import BytesIO
# from utils import draw_bounding_boxes, convert_4corners_to_x1y1x2y2
# import matplotlib.pyplot as plt


# import os
# import json
# from PIL import Image

# import torch
# import torch.nn as nn
# import torchvision.ops as ops


# from utils import (
#     convert_4corners_to_x1y1x2y2, 
#     get_features, 
#     bboxes_tointeger, 
#     compute_avg_conv_filter, 
#     rescale_tensor,
#     resize_conv_maps,
#     rescale_bbox,
#     ellipse_coverage
# )

# device = "cpu"

# def combine_pil_and_plot(original_img, density_map, title="", other_density_map=None):
#     """
#     - Convert the plot to a PIL image
#     """
#     # density_map をプロット
#     fig, ax = plt.subplots(figsize=(8, 8))
#     im = ax.imshow(density_map)
#     ax.set_title(title)
#     fig.colorbar(im, ax=ax)
#     ax.axis('off')

#     # Matplotlib図をBytesIOに保存してPIL画像に変換
#     buf = BytesIO()
#     fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
#     buf.seek(0)
#     plot_image = Image.open(buf) # < plot_imageにdensity_mapが格納

#     plt.close(fig)  # メモリ解放

#     # オリジナル画像とdensity map画像を横に結合
#     original_img = original_img.convert('RGB')
#     plot_image = plot_image.resize(original_img.size)  # サイズを合わせる

#     combined_width = original_img.width + plot_image.width
#     combined_height = max(original_img.height, plot_image.height)
#     combined = Image.new('RGB', (combined_width, combined_height), (255, 255, 255))
#     combined.paste(original_img, (0, 0))
#     combined.paste(plot_image, (original_img.width, 0))

#     buf.close()  # バッファ解放
#     return combined

# def process_example(
#     img_filename, entry, model, transform, map_keys, img_dir, density_map_dir, config, return_maps=False, gt_count=None
# ):
#     """
#     - when gt_count is not None, it is used to compute the metrics, and the density map is not used
#     """
#     # 1. 画像と密度マップの読込み
#     img = Image.open(os.path.join(img_dir, img_filename)).convert('RGB')
    
#     # 密度マップがない場合に正解のカウント数を指定することを求める条件分岐
#     # 正解のカウント数が分からないなら、numpy形式の密度マップを読み込む処理を行う
#     if density_map_dir is None:
#         assert gt_count is not None, "gt_count must be provided if density_map_dir is None"
#         #assert return_maps is False, "return_maps must be False if density_map_dir is None"
#         density_map = None
#     else:
#         if gt_count is not None:
#             print(f"Warning: gt_count is provided but density_map_dir is not None. Ignoring gt_count.")

#         density_map = np.load(os.path.join(density_map_dir, f"{img_filename.split('.')[0]}.npy"))
#     # 画像サイズから幅・高さを初期化
#     w, h = img.size    

#     # 2. 特徴量抽出
#     # backboneモデルから特徴量マップを取得する
#     with torch.no_grad():
#         feats = get_features(
#             model, img, transform, map_keys,
#             divide_et_impera=config.divide_et_impera,
#             divide_et_impera_twice=config.divide_et_impera_twice
#         )
#         # コサイン類似度／正規化を設定している場合は、L2正規化が行われる条件分岐
#         if config.cosine_similarity or config.normalize_features:
#             feats = feats / feats.norm(dim=1, keepdim=True)

#     # 3. Exemplers(バウンディングボックス)の処理
#     # アノテーションの座標形式を(x1,y1,x2,y2)に変換
#     ex_bboxes = [convert_4corners_to_x1y1x2y2(b) for b in entry['box_examples_coordinates']]
#     if config.num_exemplars is not None:
#         assert config.num_exemplars > 0, "num_exemplars must be greater than 0. config.num_exemplars = " + config.num_exemplars
#         ex_bboxes = ex_bboxes[:config.num_exemplars]
#     # 画像サイズで正規化(特徴量マップのサイズにスケール変換、特徴量マップの解像度に合わせる)
#     bboxes = np.array([(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in ex_bboxes]) * feats.shape[-1]
#     # 座標を整数に変換して、バウンディングボックス間の重なりを除去するオプションも必要に応じて設定
#     bboxes = bboxes_tointeger(bboxes, config.remove_bbox_intersection)

#     conv_maps = []
#     pooled_features_list = []
#     output_sizes = []
#     rescaled_bboxes = []
    
#     # 4. Exemplers(バウンディングボックス)ごとに処理(メインループ)
#     for bbox in bboxes:
#         bbox_tensor = torch.tensor(bbox)
#         output_size = (
#             int(bbox_tensor[3] - bbox_tensor[1]), 
#             int(bbox_tensor[2] - bbox_tensor[0])
#         )
#         # ROI Alignで例示領域の特徴量を正確に切り出す
#         pooled = ops.roi_align(
#             feats, [bbox_tensor.unsqueeze(0).float().to(device)],
#             output_size=output_size, spatial_scale=1.0
#         )
#         if config.ellipse_kernel_cleaning:
#             ellipse = ellipse_coverage(pooled.shape[-2], pooled.shape[-1]).unsqueeze(0).unsqueeze(0).to(device)
#             pooled *= ellipse
            
#         pooled_features_list.append(pooled)

#         if config.exemplar_avg:
#             continue
#         # 通常の個別フィルター
#         # 切り出した特徴量をカーネルとして使用
#         # consine_similalrityがTrueの場合はチャネル(RGB)ごとの深度別に畳み込み、Falseであれば通常の畳み込み
#         conv_weights = pooled.view(feats.shape[1], 1, *output_size)
#         conv_layer = nn.Conv2d(
#             in_channels=feats.shape[1],
#             out_channels=1 if config.cosine_similarity else feats.shape[1],
#             kernel_size=output_size,
#             padding=0,
#             groups=1 if config.cosine_similarity else feats.shape[1],
#             bias=False
#         )
#         conv_layer.weight = nn.Parameter(pooled if config.cosine_similarity else conv_weights)

#         with torch.no_grad():
#             output = conv_layer(feats[0])

#         if config.correct_bbox_resize:
#             rescaled_bbox = rescale_bbox(bbox_tensor, output, feats)
#         else:
#             rescaled_bbox = bbox_tensor

#         rescaled_bboxes.append(rescaled_bbox)

#         if config.use_roi_norm and not config.roi_norm_after_mean:
#             if config.cosine_similarity:
#                 output += 1.0
#             pooled_output = ops.roi_align(
#                 output.unsqueeze(0), [rescaled_bbox.unsqueeze(0).float().to(device)],
#                 output_size=output_size, spatial_scale=1.0
#             )
#             output = output / pooled_output.sum()

#         conv_maps.append(output)
#         output_sizes.append(output_size)
#     # 全例示の平均フィルタ: すべてのRoIを平均して、1つのConv層を作成して適用
#     if config.exemplar_avg:
#         pooled = compute_avg_conv_filter(pooled_features_list)
#         output_size = pooled.shape[1:]
#         conv_weights = pooled.view(pooled.shape[0], 1, *output_size)

#         conv_layer = nn.Conv2d(
#             in_channels=feats.shape[1],
#             out_channels=1 if config.cosine_similarity else feats.shape[1],
#             kernel_size=output_size,
#             padding=0,
#             groups=1 if config.cosine_similarity else feats.shape[1],
#             bias=False
#         )
#         conv_layer.weight = nn.Parameter(pooled.unsqueeze(0) if config.cosine_similarity else conv_weights)

#         with torch.no_grad():
#             output = conv_layer(feats[0])

#         if config.use_roi_norm and not config.roi_norm_after_mean:
#             raise NotImplementedError("ROI norm after conv_mean is not implemented for average-based filter.")

#         conv_maps.append(output)
#         output_sizes.append(output_size)

#     output = post_process_density_map(
#         conv_maps, pooled_features_list, rescaled_bboxes, output_sizes, config
#     )
#     if return_maps:
#         return density_map, output # GT密度マップと推定密度マップ
    
#     # 密度マップがある場合、密度マップのSumをGround Truth(正解のカウント数)として採用
#     if density_map is not None:
#         gt_count = density_map.sum() # GT密度マップから正解数を計算
    
#     return gt_count, output.sum().item() # (正解数、推定数)
#     # MAE, RMSEの計算に使用される

# def post_process_density_map(conv_maps, pooled_feats, bboxes, output_sizes, config):
#     # ノイズの多い類似度マップをクリーンアップし、密度の分布を強調
#     # 複数のexemplarのマップをリサイズしてから平均
#     # 正規化: 中央値より低い領域を強制的に0にして背景を除去
#     if config.use_threshold:
#         output, resize_ratios = resize_conv_maps(conv_maps)
#         output = output.mean(dim=0)
#         if config.use_minmax_norm:
#             output = rescale_tensor(output)

#         thresh = torch.median(output)
#         output[output < thresh] = 0
#         return output
    
#     # 平均後のROI正規化の場合
#     if config.use_roi_norm and config.roi_norm_after_mean:
#         output, resize_ratios = resize_conv_maps(conv_maps)
#         output = output.mean(dim=0)
#         if config.use_minmax_norm:
#             output = rescale_tensor(output)
#         # exemplar領域内での平均密度を基準に全体を正規化
#         pooled_vals = []
#         for bbox, ratio in zip(bboxes, resize_ratios):
#             scaled_bbox = torch.tensor([
#                 bbox[0] * ratio[1], bbox[1] * ratio[0],
#                 bbox[2] * ratio[1], bbox[3] * ratio[0]
#             ]).int()
#             # scaled_bbox = torch.tensor(bboxes_tointeger(scaled_bbox.unsqueeze(0), config.remove_bbox_intersection)[0])
#             output_size = (
#                 int(scaled_bbox[3] - scaled_bbox[1]),
#                 int(scaled_bbox[2] - scaled_bbox[0])
#             )
#             pooled = ops.roi_align(
#                 output.unsqueeze(0).unsqueeze(0),
#                 [scaled_bbox.unsqueeze(0).float().to(device)],
#                 output_size=output_size, spatial_scale=1.0
#             )
#             pooled_vals.append(pooled)
#         # 楕円形のマスクを使用して中心付近のみを重視して合計を計算
#         if config.ellipse_normalization:
#             norm_coeff = sum([(p[0, 0] * ellipse_coverage(p.shape[-2], p.shape[-1]).to(device)).sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
#         else:
#             norm_coeff = sum([p.sum() for p in pooled_vals]) / (len(pooled_vals) * config.scaling_coeff)
#         if config.fixed_norm_coeff is not None:
#             norm_coeff = config.fixed_norm_coeff # 係数で割ることで密度を正規化

#         output = output / norm_coeff
#         if config.filter_background is True:
#             thresh = max( [f.shape[-2] * f.shape[-1] for f in pooled_feats] )
#             thresh = (1 / thresh ) * 1.0
#             output[output < thresh] = 0

#     return output

# # from convolutional_counting import VisualBackbone, re, T, timm
# import torch.nn.functional as F
# import torchvision.transforms as T
# import re
# import timm
# from model import VisualBackbone


# model_name = 'dinov2_vitl14_reg'

# resize_dim = 840 if 'dinov2' in model_name else 480
# model = VisualBackbone(model_name, img_size=resize_dim).eval()  # .to(device)不要

# transform = T.Compose([
#     T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
#     T.ToTensor(),
#     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
# ])

# map_keys = ['vit_out']


# config = {
#     'model_name': model_name,
#     'img_dir': '.',
#     'density_map_dir': '.',
    
#     'CARPK' : False,
    
#     'divide_et_impera' : True,
#     'divide_et_impera_twice': False,
#     'normalize_features' : False, #<-----
#     'filter_background': True,
#     'ellipse_normalization': True,
#     'ellipse_kernel_cleaning' : False,
#     'exemplar_avg' : False, # as conv_filter
#     'correct_bbox_resize' : True,
#     'use_roi_norm' : True,
#     'roi_norm_after_mean' : True,
#     'use_threshold' : False,
#     'cosine_similarity': False,
#     'use_minmax_norm' : True,
#     'use_density_map': True,
#     'remove_bbox_intersection': False,
#     'scaling_coeff': 1.0,
#     'normalize_only_biggest_bbox':False,
#     'fixed_norm_coeff' : None,
#     'num_exemplars': None
# }
# # convert dict to config object
# class Config:
#     def __init__(self, **kwargs):
#         self.__dict__.update(kwargs)
# config = Config(**config)

# # 新しい画像（check.png）で推論
# filepath = '/Users/isaoishikawa/CountingDINO_flux/assets/beard1_resize.jpeg'  # ← ここをcheck.pngのパスに（同じフォルダならこれでOK）

# img = Image.open(filepath).convert('RGB')

# # Exemplarの3個のバウンディングボックス（ここをcheck.pngに合わせて手動修正！！）
# # 形式: [[左上x, 左上y], [左下x, 左下y], [右下x, 右下y], [右上x, 右上y]] = 4点形式にする：[[x1,y1], [x1,y1+h], [x1+w,y1+h], [x1+w,y1]]
# # beard1_resize.jpeg用のバウンディングボックス
# example = {'box_examples_coordinates': [
#   [[616, 929],[616, 949],[640, 949],[640, 929]], # 1個目
#   [[114, 147],[114, 168],[146, 168],[146, 147]], # 2個目
#   [[963, 52],[963, 77],[987, 77],[987, 52]] # 3個目
# ]}

# # 4点形式 → (x1, y1, x2, y2) に変換して描画
# ex_bboxes_check = [convert_4corners_to_x1y1x2y2(bbox) for bbox in example['box_examples_coordinates']]

# # 赤い太枠で描画（見やすいようにwidth=5）
# img_with_boxes = draw_bounding_boxes(
#     img, 
#     ex_bboxes_check, 
#     color="red", 
#     width=5, 
#     text_background=True
# )

# print("保存しました: check_with_boxes.png（Finderで確認もOK）")

# # 推論実行
# gt_map, pred_map = process_example(
#     os.path.basename(filepath),
#     example, 
#     model, transform, map_keys, 
#     os.path.dirname(filepath) or '.', 
#     None, 
#     config, 
#     return_maps=True,
#     gt_count=0  # ← これ追加
# )

# # 可視化
# # 4点形式 → (x1, y1, x2, y2) に変換して描画
# ex_bboxes = [convert_4corners_to_x1y1x2y2(bbox) for bbox in example['box_examples_coordinates']]
# # 赤い太枠で描画（見やすいようにwidth=5）
# img_with_boxes = draw_bounding_boxes(img, ex_bboxes, color="red", width=5)

# density_map = pred_map.cpu().numpy()
# pred = density_map.sum().item()

# to_plot_img = combine_pil_and_plot(
#     original_img=img_with_boxes, 
#     density_map=density_map, 
#     title=f"Prediction: {pred:.1f}", 
# )

# to_plot_img