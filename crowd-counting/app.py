# app_v3.py
"""
Exemplar-based Object Counting App v3
- パラメータ調整スライダー付きUI
- scaling_coeff, filter_background閾値倍率をリアルタイム調整可能
"""
import gradio as gr
from PIL import Image

from src.inference import run_inference_with_params
from src.utils import draw_bounding_boxes
from ui.components import (
    create_image_annotator,
    create_preview_image,
    create_density_only,
    create_overlay_image,
    create_count_output
)


def parse_annotation_boxes(annotation_data):
    """
    gradio_image_annotationの出力形式をparse_boxes互換形式に変換

    Input:  {"image": PIL.Image, "boxes": [{"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2, ...}, ...]}
    Output: [[x1, y1, x2, y2], ...]
    """
    if annotation_data is None:
        return [], None

    img = annotation_data.get("image")
    boxes = annotation_data.get("boxes", [])

    exemplars = []
    for box in boxes:
        x1 = box.get("xmin", 0)
        y1 = box.get("ymin", 0)
        x2 = box.get("xmax", 0)
        y2 = box.get("ymax", 0)
        if x1 < x2 and y1 < y2:
            exemplars.append([x1, y1, x2, y2])

    return exemplars, img


def update_preview(annotation_data):
    """
    リアルタイムプレビュー更新
    アノテーションデータから画像とボックスを取得し、赤枠でプレビュー
    """
    exemplars, img = parse_annotation_boxes(annotation_data)

    if img is None:
        return None

    if not exemplars:
        return img

    return draw_bounding_boxes(img.copy(), exemplars, color="red", width=5)


def run_counting(annotation_data, scaling_coeff, bg_thresh_multiplier):
    """
    メイン推論関数（パラメータ調整対応）
    """
    exemplars, img = parse_annotation_boxes(annotation_data)

    if img is None:
        return None, None, None, None, "画像をアップロードしてください", None

    if len(exemplars) == 0:
        preview = img
        return preview, None, None, None, "少なくとも1つの有効なExemplarをドラッグで囲んでください", None

    try:
        preview, density_only, overlay, grid, count = run_inference_with_params(
            img,
            exemplars,
            scaling_coeff=scaling_coeff,
            bg_thresh_multiplier=bg_thresh_multiplier
        )
        return preview, density_only, overlay, grid, f"**推定カウント: {count:.1f}**", count
    except Exception as e:
        preview = update_preview(annotation_data)
        return preview, None, None, preview, f"エラーが発生しました: {str(e)}", None


def create_example_annotation(image_path, boxes):
    """
    Examples用のアノテーションデータを作成
    boxes: [[x, y, w, h], ...] (x,y,幅,高さ形式)
    """
    exemplars = []
    for box in boxes:
        if len(box) == 4:
            x, y, w, h = box
            x1, y1, x2, y2 = x, y, x + w, y + h
            exemplars.append({
                "xmin": int(x1),
                "ymin": int(y1),
                "xmax": int(x2),
                "ymax": int(y2),
                "label": "Exemplar",
                "color": (255, 0, 0)
            })
    return {
        "image": image_path,
        "boxes": exemplars
    }


# メインUI構築
with gr.Blocks(title="Exemplar-based Counting v3 (Parameter Tuning)") as demo:
    gr.Markdown("# Exemplar-based Object Counting v3")
    gr.Markdown("""
    ## 使い方
    1. **画像をアップロード**
    2. **Exemplarを指定**: 画像上でドラッグして、カウントしたい対象を矩形で囲みます
    3. **パラメータを調整**: スライダーで調整（過検出時は値を上げる）
    4. **Run Counting** ボタンをクリック
    """)

    with gr.Row():
        with gr.Column(scale=1):
            # 画像アノテーター（ドラッグで矩形描画）
            annotator = create_image_annotator(
                label_list=["Exemplar"],
                label_colors=[(255, 0, 0)]
            )

        with gr.Column(scale=1):
            preview_image = create_preview_image()

    # パラメータ調整セクション
    gr.Markdown("### パラメータ調整")
    with gr.Row():
        with gr.Column():
            scaling_coeff = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.05,
                label="Scaling Coefficient（正規化係数）",
                info="過検出の場合は上げる（1.1〜1.3）、過少検出の場合は下げる（0.8〜0.95）"
            )
        with gr.Column():
            bg_thresh_multiplier = gr.Slider(
                minimum=0.5,
                maximum=5.0,
                value=1.0,
                step=0.1,
                label="背景閾値倍率",
                info="背景ノイズが多い場合は上げる（1.5〜3.0）"
            )

    # 調整ガイド
    with gr.Accordion("パラメータ調整ガイド", open=False):
        gr.Markdown("""
        ### Scaling Coefficient（正規化係数）
        - **デフォルト**: 1.0
        - **過検出（推定 > 正解）の場合**: 値を上げる
          - 乖離5%程度 → 1.05〜1.1
          - 乖離10%程度 → 1.1〜1.15
          - 乖離20%程度 → 1.2〜1.25
          - 乖離30%程度 → 1.3〜1.4
        - **過少検出（推定 < 正解）の場合**: 値を下げる（0.8〜0.95）

        ### 背景閾値倍率
        - **デフォルト**: 1.0
        - 背景に髭と似たテクスチャがあり誤検出が多い場合は上げる
        - 細い/薄い髭が検出されない場合は下げる
        """)

    run_btn = gr.Button("Run Counting", variant="primary", size="lg")

    # 結果表示エリア
    with gr.Row():
        with gr.Column():
            density_only_img = create_density_only()
        with gr.Column():
            overlay_img = create_overlay_image()

    with gr.Row():
        status_text = gr.Markdown()
        count_output = create_count_output()

    # リアルタイムプレビュー更新（アノテーション変更時）
    annotator.change(
        fn=update_preview,
        inputs=[annotator],
        outputs=[preview_image]
    )

    # 実行ボタンクリック時
    run_btn.click(
        fn=run_counting,
        inputs=[annotator, scaling_coeff, bg_thresh_multiplier],
        outputs=[preview_image, density_only_img, overlay_img, preview_image, status_text, count_output]
    )

    # サンプル（beard1_resize.jpeg用）
    example_data = create_example_annotation(
        "assets/beard1_resize.jpeg",
        [[616, 929, 24, 20], [114, 147, 32, 21], [963, 52, 24, 25]]  # x, y, w, h形式
    )

    gr.Examples(
        examples=[[example_data]],
        inputs=[annotator],
        label="サンプル（クリックで読み込み）"
    )


if __name__ == "__main__":
    demo.launch(share=True)
