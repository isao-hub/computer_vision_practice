# components.py
import gradio as gr

def create_image_input():
    return gr.Image(label="画像をアップロード", type="pil", height=500)

def create_box_input():
    return gr.Textbox(
        label="Exemplarバウンディングボックス（複数行可）",
        placeholder="形式1（推奨）: x,y,幅,高さ\n"
                    "形式2: x1,y1,x2,y2\n"
                    "例:\n"
                    "100,100,80,80\n"
                    "200,150,90,75\n"
                    "300,300,100,100",
        lines=6,
        info="1行に1つのExemplar。少なくとも1つは入力してください。"
    )

def create_preview_image():
    return gr.Image(type="pil", label="プレビュー（指定されたExemplarを赤枠で表示）", height=500)

def create_output_image():
    return gr.Image(type="pil", label="結果: 入力+BBox (左) / Density Map (右)", height=550)

def create_count_output():
    return gr.Number(label="推定カウント数", precision=1)