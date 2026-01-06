"""Wrapper for simple-lama-inpainting with batch processing support."""

import numpy as np
from PIL import Image
from typing import Dict, Optional, List, Tuple
from simple_lama_inpainting import SimpleLama

# 定義した関数はimportできる
# ※Pythonではフォルダを「パッケージ」として扱うために、各フォルダに__init__.pyファイルが必要となる。このファイルがあることで、Pythonが該当フォルダを「単なるディレクトリ」ではなく、「インポート可能なパッケージ」として認識する
from core.image_utils import numpy_to_pil, pil_to_numpy, validate_image_and_mask


class InpaintingEngine:
    """Wrapper for simple-lama-inpainting with caching and batch processing."""

    def __init__(self):
        """Initialize the inpainting engine."""
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the SimpleLama model."""
        try:
            self.model = SimpleLama()
        except Exception as e:
            raise RuntimeError(f"SimpleLamaモデルの初期化に失敗しました: {str(e)}")

    def inpaint(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Perform inpainting on a single image with a mask.

        Args:
            image: Input PIL Image (RGB)
            mask: Mask PIL Image (L mode, white=inpaint area)

        Returns:
            Inpainted PIL Image
        """
        is_valid, error_msg = validate_image_and_mask(image, mask)
        if not is_valid:
            raise ValueError(error_msg)

        if mask.mode != 'L':
            mask = mask.convert('L')

        if image.mode != 'RGB':
            image = image.convert('RGB')

        try:
            result = self.model(image, mask)
            return result
        except Exception as e:
            raise RuntimeError(f"Inpainting処理に失敗しました: {str(e)}")

    def batch_inpaint_stages(
        self,
        original_image: Image.Image,
        stage_masks: Dict[int, np.ndarray]
    ) -> Dict[int, Image.Image]:
        """
        Process multiple stages sequentially, each from the original image.

        Args:
            original_image: Original PIL Image
            stage_masks: Dictionary mapping stage percentage to mask array

        Returns:
            Dictionary mapping stage percentage to inpainted PIL Image
        """
        results = {}

        for percentage in sorted(stage_masks.keys()):
            mask_array = stage_masks[percentage]

            if mask_array is None:
                continue

            mask_pil = numpy_to_pil(mask_array)

            try:
                inpainted = self.inpaint(original_image, mask_pil)
                results[percentage] = inpainted
            except Exception as e:
                print(f"Stage {percentage}% の処理に失敗しました: {str(e)}")
                continue

        return results

    def is_initialized(self) -> bool:
        """Check if the model is properly initialized."""
        return self.model is not None


class ProgressiveInpainter:
    """High-level interface for progressive inpainting."""

    def __init__(self):
        """Initialize the progressive inpainter."""
        self.engine = InpaintingEngine()

    def process_progressive(
        self,
        image: Image.Image,
        stage_masks: Dict[int, np.ndarray],
        progress_callback: Optional[callable] = None
    ) -> Tuple[Dict[int, Image.Image], List[str]]:
        """
        Process progressive inpainting with optional progress callback.

        Args:
            image: Original PIL Image
            stage_masks: Dictionary mapping stage percentage to mask array
            progress_callback: Optional callback function(current, total, stage)

        Returns:
            Tuple of (results dict, status messages list)
        """
        results = {}
        messages = []

        total_stages = len(stage_masks)
        current_stage = 0

        for percentage in sorted(stage_masks.keys()):
            current_stage += 1

            if progress_callback:
                progress_callback(current_stage, total_stages, percentage)

            mask_array = stage_masks[percentage]

            if mask_array is None:
                messages.append(f"Stage {percentage}%: マスクがありません")
                continue

            mask_pil = numpy_to_pil(mask_array)

            try:
                inpainted = self.engine.inpaint(image, mask_pil)
                results[percentage] = inpainted
                messages.append(f"Stage {percentage}%: 完了")
            except Exception as e:
                error_msg = f"Stage {percentage}%: エラー - {str(e)}"
                messages.append(error_msg)
                print(error_msg)

        return results, messages
