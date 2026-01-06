"""Gradio UI components for the progressive inpainting application."""

import gradio as gr
import config


def create_input_panel():
    """
    Create the input panel with image upload, mask drawing, and controls.

    Returns:
        Dictionary of component references
    """
    components = {}

    with gr.Column(scale=1):
        gr.Markdown("## 入力")
        # type: filepath, numpyなども指定できるが、simple_lama_inpaintingの引数にはPIL.Image型を想定しているかつ処理のしやすさからpilを指定
        # sources: webcam, clipboardなども指定できるが、手持ちのファイルから任意の画像を選べれば良いのでuploadを指定
        components['image_upload'] = gr.Image(
            label="画像をアップロード",
            type="pil",
            sources=["upload"],
            image_mode="RGB"
        )

        components['image_editor'] = gr.ImageEditor(
            label="マスクを描画（削除したい物体を白で塗りつぶしてください）",
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

        components['stage_checkboxes'] = gr.CheckboxGroup(
            choices=config.DEFAULT_STAGES,
            value=config.DEFAULT_STAGES,
            label="ステージ（%）",
            info="どのステージを生成するか選択してください"
        )

        components['process_btn'] = gr.Button(
            "段階的削除を実行",
            variant="primary",
            size="lg"
        )
    # 辞書型{'stage_checkboxes': gr.CheckboxGroup(),'process_btn': gr.Button()}のcomponentsを最終結果とする
    return components


def create_output_panel():
    """
    Create the output panel with gallery and status display.

    Returns:
        Dictionary of component references
    """
    components = {}

    with gr.Column(scale=2):
        gr.Markdown("## 結果")

        components['gallery'] = gr.Gallery(
            label="段階的削除の結果",
            columns=3,
            rows=2,
            height="auto",
            object_fit="contain",
            show_label=True,
            preview=True
        )

        components['status'] = gr.Textbox(
            label="ステータス",
            value="画像をアップロードしてマスクを描画してください",
            interactive=False,
            lines=3
        )

        components['progress'] = gr.Progress()
    # 辞書型{'gallery': gr.Gallery(),'status': gr.Textbox()}のcomponentsを最終結果とする
    return components


def create_info_panel():
    """
    Create an informational panel with instructions.

    Returns:
        Markdown component
    """
    info_text = """
### 使い方

1. **画像をアップロード**: 物体を削除したい画像を選択
2. **マスクを描画**: ImageEditorで削除したい物体を白で塗りつぶす
3. **マスクを追加**: 描画したマスクを保存（各マスク = 1つの物体）
4. **繰り返し**: 削除したい物体ごとに手順2-3を繰り返す
5. **ステージ選択**: 生成したいステージ（%）を選択
6. **実行**: 「段階的削除を実行」ボタンをクリック

### 例
- 10個の物体（マスク）がある場合:
  - **20%**: 2個削除（8個残存）
  - **40%**: 4個削除（6個残存）
  - **60%**: 6個削除（4個残存）
  - **80%**: 8個削除（2個残存）
  - **100%**: 全削除

### 技術
- **LaMa Inpainting**: フーリエ畳み込みを使った高品質な画像修復
- **段階的処理**: 各ステージは元画像から独立して処理されます
"""

    return gr.Markdown(info_text)


def create_example_section():
    """
    Create an examples section (placeholder for future use).

    Returns:
        Markdown component
    """
    example_text = """
### サンプル画像
サンプル画像を使って機能を試すことができます。
"""

    return gr.Markdown(example_text, visible=False)
