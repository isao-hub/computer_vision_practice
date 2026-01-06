"""Mask aggregation and stage calculation logic."""

import numpy as np
from typing import List, Dict, Optional
from PIL import Image

# 定義した関数はimportできる
# ※Pythonではフォルダを「パッケージ」として扱うために、各フォルダに__init__.pyファイルが必要となる。このファイルがあることで、Pythonが該当フォルダを「単なるディレクトリ」ではなく、「インポート可能なパッケージ」として認識する
from core.image_utils import merge_masks, convert_to_binary_mask, has_mask_content


class MaskLayer:
    """Represents a single mask drawn by user."""
    # classを定義するときには__init__関数の定義が必須
    def __init__(self, mask_array: np.ndarray, index: int):
        """
        Initialize a mask layer.

        Args:
            mask_array: Binary mask array
            index: Layer index
        """
        self.mask_array = convert_to_binary_mask(mask_array)
        self.index = index

    def to_binary_mask(self) -> np.ndarray:
        """Get the binary mask array."""
        return self.mask_array

    def get_bounds(self) -> tuple:
        """
        Get the bounding box of the mask.

        Returns:
            Tuple of (x, y, width, height) or None if mask is empty
        """
        if not has_mask_content(self.mask_array):
            return None

        rows = np.any(self.mask_array > 0, axis=1)
        cols = np.any(self.mask_array > 0, axis=0)

        if not np.any(rows) or not np.any(cols):
            return None

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        return (x_min, y_min, width, height)


class MaskAggregator:
    """Manages multiple mask layers and stage-wise aggregation."""

    def __init__(self):
        """Initialize the mask aggregator."""
        self.mask_layers: List[MaskLayer] = []

    def add_mask_layer(self, mask: np.ndarray) -> int:
        """
        Add a new mask layer.

        Args:
            mask: Mask array to add

        Returns:
            Index of the added mask layer
        """
        if mask is None:
            raise ValueError("マスクがNoneです")

        if not has_mask_content(mask):
            raise ValueError("マスクに内容がありません")

        index = len(self.mask_layers)
        mask_layer = MaskLayer(mask, index)
        self.mask_layers.append(mask_layer)

        return index

    def get_cumulative_mask(self, up_to_index: int) -> Optional[np.ndarray]:
        """
        Get cumulative mask up to specified index.

        Args:
            up_to_index: Index up to which to merge masks (inclusive)

        Returns:
            Merged mask array or None if no masks
        """
        if up_to_index < 0 or up_to_index >= len(self.mask_layers):
            return None

        masks_to_merge = [
            layer.to_binary_mask()
            for layer in self.mask_layers[:up_to_index + 1]
        ]

        return merge_masks(masks_to_merge)

    def calculate_stage_masks(self, stage_percentages: List[int]) -> Dict[int, np.ndarray]:
        """
        Calculate masks for each stage based on percentages.

        Args:
            stage_percentages: List of percentages (e.g., [20, 40, 60, 80, 100])

        Returns:
            Dictionary mapping percentage to cumulative mask
        """
        total_masks = len(self.mask_layers)

        if total_masks == 0:
            return {}

        stage_masks = {}

        for percentage in sorted(stage_percentages):
            mask_count = max(1, int(total_masks * percentage / 100))
            mask_count = min(mask_count, total_masks)

            cumulative_mask = self.get_cumulative_mask(mask_count - 1)
            stage_masks[percentage] = cumulative_mask

        return stage_masks

    def get_total_mask_count(self) -> int:
        """Get the total number of mask layers."""
        return len(self.mask_layers)
    # "すべてクリア"ボタンで処理される
    def clear_all_masks(self):
        """Remove all mask layers."""
        self.mask_layers = []

    def remove_last_mask(self) -> bool:
        """
        Remove the most recently added mask.

        Returns:
            True if a mask was removed, False if no masks exist
        """
        if len(self.mask_layers) > 0:
            self.mask_layers.pop()
            return True
        return False

    def get_mask_layer(self, index: int) -> Optional[MaskLayer]:
        """
        Get a specific mask layer by index.

        Args:
            index: Index of the mask layer

        Returns:
            MaskLayer or None if index is invalid
        """
        if 0 <= index < len(self.mask_layers):
            return self.mask_layers[index]
        return None


def calculate_stage_indices(total_masks: int, stage_percentages: List[int]) -> Dict[int, int]:
    """
    Calculate the number of masks to use for each stage.

    Args:
        total_masks: Total number of available masks
        stage_percentages: List of percentages (e.g., [20, 40, 60, 80, 100])

    Returns:
        Dictionary mapping percentage to number of masks
        Example: 37 masks, [20, 40, 60, 80, 100] -> {20: 7, 40: 15, 60: 22, 80: 30, 100: 37}
    """
    if total_masks == 0:
        return {}

    stage_indices = {}

    for pct in stage_percentages:
        mask_count = max(1, int(total_masks * pct / 100))
        mask_count = min(mask_count, total_masks)
        stage_indices[pct] = mask_count

    return stage_indices
