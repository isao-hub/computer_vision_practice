"""Gradio UI components for the beard thinning application."""

import gradio as gr
import config


def create_input_panel():
    """
    入力パネルを作成する（画像アップロード、マスク描画、コントロール）。

    Returns:
        コンポーネント参照の辞書
    """
    components = {}

    with gr.Column(scale=1):
        gr.Markdown("## 入力")

        components['image_upload'] = gr.Image(
            label="顔画像をアップロード",
            type="pil",
            sources=["upload"],
            image_mode="RGB"
        )

        components['image_editor'] = gr.ImageEditor(
            label="髭の領域をマスク（白で塗りつぶしてください）",
            type="numpy",
            brush=gr.Brush(
                default_size=config.BRUSH_RADIUS_DEFAULT,
                colors=["white"],
                default_color="white"
            ),
            eraser=gr.Eraser(default_size=config.BRUSH_RADIUS_DEFAULT),
            image_mode="RGB"
        )

        with gr.Row():
            components['add_mask_btn'] = gr.Button(
                "マスクを追加",
                variant="secondary",
                size="sm"
            )
            components['clear_masks_btn'] = gr.Button(
                "すべてクリア",
                variant="secondary",
                size="sm"
            )

        components['mask_counter'] = gr.Textbox(
            label="追加されたマスク数",
            value="0個のマスク",
            interactive=False
        )

        gr.Markdown("---")

        components['thinning_checkboxes'] = gr.CheckboxGroup(
            choices=config.DEFAULT_THINNING_LEVELS,
            value=config.DEFAULT_THINNING_LEVELS,
            label="薄め具合（%）",
            info="どの薄さレベルを生成するか選択してください"
        )

        components['process_btn'] = gr.Button(
            "髭薄めを実行",
            variant="primary",
            size="lg"
        )

    return components


def create_output_panel():
    """
    出力パネルを作成する（ギャラリーとステータス表示）。

    Returns:
        コンポーネント参照の辞書
    """
    components = {}

    with gr.Column(scale=2):
        gr.Markdown("## 結果")

        components['gallery'] = gr.Gallery(
            label="髭薄めの結果",
            columns=3,
            rows=2,
            height="auto",
            object_fit="contain",
            show_label=True,
            preview=True
        )

        components['status'] = gr.Textbox(
            label="ステータス",
            value="画像をアップロードして髭の領域をマスクしてください",
            interactive=False,
            lines=3
        )

    return components


def create_info_panel():
    """
    説明パネルを作成する。

    Returns:
        Markdownコンポーネント
    """
    info_text = """
### 使い方

1. **画像をアップロード**: 男性の顔画像を選択
2. **髭をマスク**: ImageEditorで髭の領域を白で塗りつぶす
3. **マスクを追加**: 描画したマスクを保存
4. **薄め具合を選択**: 生成したいレベル（30%, 50%, 70%, 100%）を選択
5. **実行**: 「髭薄めを実行」ボタンをクリック

### 髭薄めの仕組み

- **LaMa Inpainting**: フーリエ畳み込みを使った高品質な画像修復で髭を完全除去
- **Alpha Blending**: 元画像と髭除去画像をブレンドして段階的な薄め効果を実現
  - **30% 薄め**: 元の髭の70%が残存
  - **50% 薄め**: 元の髭の50%が残存
  - **70% 薄め**: 元の髭の30%が残存
  - **100%**: 完全に髭を除去

### 脱毛経過観察への活用

髭脱毛の効果をシミュレーションし、将来の経過を可視化できます。
"""

    return gr.Markdown(info_text)
