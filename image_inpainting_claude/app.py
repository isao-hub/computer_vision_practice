"""Main Gradio application for progressive image inpainting."""

import gradio as gr
# 定義した関数はimportできる
# ※Pythonではフォルダを「パッケージ」として扱うために、各フォルダに__init__.pyファイルが必要となる。このファイルがあることで、Pythonが該当フォルダを「単なるディレクトリ」ではなく、「インポート可能なパッケージ」として認識する
from ui.components import create_input_panel, create_output_panel, create_info_panel
from ui.callbacks import (
    AppState,
    on_image_upload,
    on_add_mask_click,
    on_clear_masks_click,
    on_process_click,
    get_mask_counter_text
)


def build_app():
    """
    Build and configure the Gradio application.

    Returns:
        Gradio Blocks application
    """
    with gr.Blocks(
        title="Progressive Image Inpainting",
        theme=gr.themes.Soft()
    ) as app:
        gr.Markdown(
            """
            # 段階的物体削除 - Progressive Object Removal

            Google「消しゴムマジック」のように物体を削除し、段階的な消去プロセスを可視化します。
            """
        )

        app_state = gr.State(AppState())

        with gr.Row():
            with gr.Column(scale=1):
                input_components = create_input_panel()
                info_panel = create_info_panel()

            with gr.Column(scale=2):
                output_components = create_output_panel()
        # componentsの初期化
        image_upload = input_components['image_upload']
        image_editor = input_components['image_editor']
        add_mask_btn = input_components['add_mask_btn']
        clear_masks_btn = input_components['clear_masks_btn']
        mask_counter = input_components['mask_counter']
        stage_checkboxes = input_components['stage_checkboxes']
        process_btn = input_components['process_btn']

        gallery = output_components['gallery']
        status = output_components['status']
        # componentsとイベントハンドラー(関数の処理)の紐付け: fn(function)、inputs(引数)、outputs(functionの最終出力)で整合させる
        image_upload.change(
            fn=on_image_upload,
            inputs=[image_upload, app_state],
            outputs=[image_editor, status]
        )

        add_mask_btn.click(
            fn=on_add_mask_click,
            inputs=[image_editor, app_state],
            outputs=[app_state, mask_counter]
        )

        clear_masks_btn.click(
            fn=on_clear_masks_click,
            inputs=[app_state],
            outputs=[app_state, mask_counter]
        )

        process_btn.click(
            fn=on_process_click,
            inputs=[image_upload, stage_checkboxes, app_state],
            outputs=[gallery, status]
        )

        gr.Markdown(
            """
            ---
            ### 技術スタック
            - **LaMa Inpainting**: フーリエ畳み込みによる高品質画像修復
            - **Gradio**: インタラクティブなWebインターフェース
            - **Python**: バックエンド処理

            [GitHub](https://github.com/enesmsahin/simple-lama-inpainting) |
            [LaMa Paper](https://github.com/advimman/lama)
            """
        )

    return app


def main():
    """Main entry point for the application."""
    app = build_app()

    app.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )

# Pythonではすべてのファイルに__name__という変数が付与される。このファイルがターミナルで実行された場合: python app.py、__name__の値が__main__に切り替わる。そのためpython app.pyを実行したら、main()処理を実行することを意図したコードを用意する。
if __name__ == "__main__":
    main()
