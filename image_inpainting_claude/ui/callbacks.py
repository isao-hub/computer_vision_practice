"""Event handlers and callbacks for the Gradio interface."""

import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional
import gradio as gr

# 定義した関数はimportできる
# ※Pythonではフォルダを「パッケージ」として扱うために、各フォルダに__init__.pyファイルが必要となる。このファイルがあることで、Pythonが該当フォルダを「単なるディレクトリ」ではなく、「インポート可能なパッケージ」として認識する
from core.mask_processor import MaskAggregator
from core.inpainting import ProgressiveInpainter
from core.image_utils import (
    extract_mask_from_editor,
    has_mask_content,
    resize_image_if_needed,
    numpy_to_pil
)


class AppState:
    """Manages application state across interactions."""
    # self = AppStateで、mask_aggregater, inpainter, original_imageの3オブジェクトを事前定義したクラスなどで初期化
    def __init__(self):
        self.mask_aggregator = MaskAggregator()
        self.inpainter = ProgressiveInpainter()
        self.original_image: Optional[Image.Image] = None

    def reset(self):
        """Reset the application state."""
        self.mask_aggregator.clear_all_masks()
        self.original_image = None


def on_image_upload(image: Image.Image, state: AppState) -> Tuple[Image.Image, str]:
    """
    Handle image upload event.

    Args:
        image: Uploaded PIL Image
        state: Application state

    Returns:
        Tuple of (processed image, status message)
    """
    if image is None:
        return None, "画像が選択されていません"

    state.original_image = resize_image_if_needed(image)
    # 出力はTuple型＝(画像, 文字列)のペアの順番で固定する
    return state.original_image, f"画像をアップロードしました（サイズ: {state.original_image.size}）"

# 「追加されたマスク数」での処理
def on_add_mask_click(
    editor_output: dict,
    state: AppState
) -> Tuple[Dict, str]:
    """
    Handle mask addition event.

    Args:
        editor_output: Output from ImageEditor
        state: Application state

    Returns:
        Tuple of (updated state dict, status message)
    """
    try:
        mask = extract_mask_from_editor(editor_output)
        # 「マスクを描画」にて画像をアップせずに「マスクを追加」した場合の処理
        if mask is None:
            return state, "マスクが検出されませんでした。ImageEditorで領域を描画してください。"
        # 「マスクを描画」にて画像をアップしたがマスクをかけず＝白で塗り潰さずに「マスクを追加」した場合の処理
        if not has_mask_content(mask):
            return state, "マスクに内容がありません。領域を描画してください。"

        state.mask_aggregator.add_mask_layer(mask)

        mask_count = state.mask_aggregator.get_total_mask_count()
        return state, f"マスクを追加しました！ 現在: {mask_count}個のマスク"

    except Exception as e:
        return state, f"エラー: {str(e)}"


def on_clear_masks_click(state: AppState) -> Tuple[Dict, str]:
    """
    Handle clear all masks event.

    Args:
        state: Application state

    Returns:
        Tuple of (updated state dict, status message)
    """
    state.mask_aggregator.clear_all_masks()
    return state, "すべてのマスクをクリアしました"

# ステージ(%)のチェックボックスと「段階的削除を実行」ボタンを押下した時の処理
def on_process_click(
    image: Image.Image,
    stage_percentages: List[int],
    state: AppState,
    progress: gr.Progress = gr.Progress()
) -> Tuple[List[Tuple[Image.Image, str]], str]:
    """
    Handle progressive inpainting execution.

    Args:
        image: Original image
        stage_percentages: Selected stage percentages
        state: Application state
        progress: Gradio progress tracker

    Returns:
        Tuple of (gallery images with captions, status message)
    """
    try:
        if image is None:
            return [], "エラー: 画像をアップロードしてください"

        state.original_image = resize_image_if_needed(image)

        mask_count = state.mask_aggregator.get_total_mask_count()

        if mask_count == 0:
            return [], "エラー: 少なくとも1つのマスクを追加してください"

        if not stage_percentages or len(stage_percentages) == 0:
            return [], "エラー: 少なくとも1つのステージを選択してください"

        progress(0, desc="ステージを計算中...")

        stage_masks = state.mask_aggregator.calculate_stage_masks(stage_percentages)

        if not stage_masks:
            return [], "エラー: ステージマスクの計算に失敗しました"

        progress(0.1, desc="Inpainting処理を開始...")

        def progress_callback(current, total, stage):
            progress_value = 0.1 + (0.9 * current / total)
            progress(progress_value, desc=f"Stage {stage}% を処理中... ({current}/{total})")

        results, messages = state.inpainter.process_progressive(
            state.original_image,
            stage_masks,
            progress_callback=progress_callback
        )

        if not results:
            return [], "エラー: Inpainting処理に失敗しました\n" + "\n".join(messages)

        gallery_images = []
        for percentage in sorted(results.keys()):
            result_image = results[percentage]
            caption = f"Stage {percentage}% ({int(mask_count * percentage / 100)}/{mask_count} objects removed)"
            gallery_images.append((result_image, caption))

        status_msg = f"完了！ {len(results)}個のステージを生成しました\n"
        status_msg += f"総マスク数: {mask_count}\n"
        status_msg += "\n".join(messages)

        progress(1.0, desc="完了！")

        return gallery_images, status_msg

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error in on_process_click: {error_detail}")
        return [], f"エラーが発生しました: {str(e)}"


def get_mask_counter_text(state: AppState) -> str:
    """
    Get the mask counter display text.

    Args:
        state: Application state

    Returns:
        Counter text
    """
    count = state.mask_aggregator.get_total_mask_count()
    return f"{count}個のマスク"
