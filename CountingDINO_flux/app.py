# app.py
import gradio as gr
from PIL import Image
import re
import numpy as np
from src.inference import run_inference
from src.utils import draw_bounding_boxes
from ui.components import (
    create_image_input, create_box_input,
    create_preview_image, create_output_image, create_count_output
)

# Textboxの入力テキストをパースして [[x1,y1,x2,y2], ...] に変換
def parse_boxes(text: str):
    if not text:
        return []
    exemplars = []
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    for line in lines:
        # カンマ区切りで4つの数値を取得
        nums = re.split(r'[,;\s]+', line.strip())
        if len(nums) != 4:
            continue
        try:
            vals = [float(x) for x in nums]
            if vals[2] > 1 and vals[3] > 1:  # width/height が与えられた場合
                x, y, w, h = vals
                x1, y1, x2, y2 = x, y, x + w, y + h
            else:  # x1,y1,x2,y2 の場合
                x1, y1, x2, y2 = vals
            if x1 >= x2 or y1 >= y2:
                continue
            exemplars.append([x1, y1, x2, y2])
        except:
            continue
    return exemplars

# [カスタマイズ] 出力用に新しいコンポーネントを追加
def create_density_only():
    return gr.Image(type="pil", label="密度マップ (viridis colormap)", height=500)

def create_overlay_image():
    return gr.Image(type="pil", label="密度マップを画像に重ね合わせ (半透明)", height=500)

# プレビュー生成（赤枠付き画像）
def update_preview(img_pil: Image.Image, box_text: str):
    if img_pil is None:
        return None
    exemplars = parse_boxes(box_text)
    if not exemplars:
        return img_pil
    return draw_bounding_boxes(img_pil.copy(), exemplars, color="red", width=5)

# [カスタマイズ]メイン推論関数
def run_counting(img_pil: Image.Image, box_text: str):
    if img_pil is None:
        return None, None, None, None, "画像をアップロードしてください", None

    exemplars = parse_boxes(box_text)
    if len(exemplars) == 0:
        preview = img_pil
        return preview, None, None, None, "少なくとも1つの有効なExemplar座標を入力してください", None
    try:
        preview, density_only, overlay, grid, count = run_inference(img_pil, exemplars)
        return preview, density_only, overlay, grid, f"**推定カウント: {count:.1f}**", count
    # try:
    #     preview, density_only, overlay, count = run_inference(img_pil, exemplars)
    #     return preview, density_only, overlay, preview, f"**推定カウント: {count:.1f}**", count
    except Exception as e:
        preview = update_preview(img_pil, box_text)
        return preview, None, None, preview, f"エラーが発生しました: {str(e)}", None

# [カスタマイズ]UI
with gr.Blocks(title="Exemplar-based Counting (viridis + Overlay)") as demo:
    gr.Markdown("# Exemplar-based Object Counting")
    gr.Markdown("""
    - 画像をアップロード  
    - Exemplar座標をテキストボックスに1行ずつ入力（x,y,w,h または x1,y1,x2,y2）  
    - 密度マップは **viridis** カラーで表示  
    - さらに **元の画像に密度マップを重ねたオーバーレイ** も出力されます
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = create_image_input()
            box_input = create_box_input()

        with gr.Column(scale=1):
            preview_image = create_preview_image()

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

    # リアルタイムプレビュー
    input_image.change(fn=update_preview, inputs=[input_image, box_input], outputs=preview_image)
    box_input.change(fn=update_preview, inputs=[input_image, box_input], outputs=preview_image)

    # 実行
    run_btn.click(
        fn=run_counting,
        inputs=[input_image, box_input],
        outputs=[preview_image, density_only_img, overlay_img, preview_image, status_text, count_output]
    )

    # 例（beard1_resize.jpeg 用）
    gr.Examples(
        examples=[
            [
                "assets/beard1_resize.jpeg",
                "616,929,24,20\n114,147,32,21\n963,52,24,25"
            ],
        ],
        inputs=[input_image, box_input]
    )

if __name__ == "__main__":
    demo.launch(share=True)

# Exemplar(Bounding Box)
# 616,929,24,20 
# 114,147,32,21
# 963,52,24,25