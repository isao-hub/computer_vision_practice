# components_v2.py
import gradio as gr
from gradio_image_annotation import image_annotator


def create_image_annotator(label_list=None, label_colors=None):
    """
    画像アップロード + バウンディングボックス描画コンポーネント
    ユーザーがドラッグで矩形を描画できる
    """
    return image_annotator(
        value=None,
        label="画像をアップロードし、Exemplarをドラッグで囲んでください",
        label_list=label_list or ["Exemplar"],
        label_colors=label_colors or [(255, 0, 0)],  # 赤色
        box_min_size=5,
        handle_size=8,
        box_thickness=3,
        box_selected_thickness=5,
        single_box=False,  # 複数のExemplar許可
        disable_edit_boxes=False,
        show_remove_button=True,
        image_type="pil",  # PILイメージで返す
        height=500
    )


def create_preview_image():
    """プレビュー画像（赤枠付き）"""
    return gr.Image(type="pil", label="プレビュー（指定されたExemplarを赤枠で表示）", height=500)


def create_density_only():
    """密度マップ出力"""
    return gr.Image(type="pil", label="密度マップ (viridis colormap)", height=500)


def create_overlay_image():
    """オーバーレイ画像出力"""
    return gr.Image(type="pil", label="密度マップを画像に重ね合わせ (半透明)", height=500)


def create_count_output():
    """カウント数出力"""
    return gr.Number(label="推定カウント数", precision=1)
