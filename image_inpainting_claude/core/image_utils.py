"""Utility functions for image and mask processing."""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import config


def merge_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine multiple binary masks using OR operation.

    Args:
        masks: List of binary mask arrays (values should be 0 or 255)

    Returns:
        Single merged mask as numpy array
    """
    if not masks:
        return None

    # 0の配列にして黒背景のマスクイメージとする
    merged = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        merged = np.maximum(merged, mask)

    return merged


def convert_to_binary_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    """
    Convert any mask format to binary (0 or 255).

    Args:
        mask: Input mask array
        threshold: Threshold value for binarization (default: 127)

    Returns:
        Binary mask with values 0 or 255
    """
    if len(mask.shape) == 3:
        mask = np.mean(mask, axis=2)

    binary_mask = np.where(mask > threshold, 255, 0).astype(np.uint8)
    return binary_mask

# アップロードされた画像サイズの前処理後のサイズを表示する際に利用する
def resize_image_if_needed(image: Image.Image, max_size: int = None) -> Image.Image:
    """
    Resize image while maintaining aspect ratio if it exceeds max_size.

    Args:
        image: PIL Image to resize
        max_size: Maximum dimension (width or height). Uses config value if None.

    Returns:
        Resized PIL Image
    """
    if max_size is None:
        max_size = config.MAX_IMAGE_SIZE

    width, height = image.size

    if width <= max_size and height <= max_size:
        return image

    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def validate_image_and_mask(image: Image.Image, mask: Image.Image) -> Tuple[bool, Optional[str]]:
    """
    Ensure image and mask have compatible dimensions.

    Args:
        image: Input PIL Image
        mask: Mask PIL Image

    Returns:
        Tuple of (is_valid, error_message)
    """
    if image is None:
        return False, "画像が指定されていません"

    if mask is None:
        return False, "マスクが指定されていません"

    if image.size != mask.size:
        return False, f"画像とマスクのサイズが一致しません。画像: {image.size}, マスク: {mask.size}"

    return True, None


def extract_mask_from_editor(editor_output: dict) -> Optional[np.ndarray]:
    """
    Extract mask from Gradio ImageEditor output.

    Args:
        editor_output: Dictionary output from Gradio ImageEditor

    Returns:
        Mask as numpy array or None if no mask present
    """
    if editor_output is None:
        return None

    if isinstance(editor_output, dict):
        if 'layers' in editor_output and len(editor_output['layers']) > 0:
            mask_layer = editor_output['layers'][0]
            if isinstance(mask_layer, np.ndarray):
                return convert_to_binary_mask(mask_layer)

        if 'composite' in editor_output:
            composite = editor_output['composite']
            if isinstance(composite, np.ndarray):
                return convert_to_binary_mask(composite)

    if isinstance(editor_output, np.ndarray):
        return convert_to_binary_mask(editor_output)

    return None


def numpy_to_pil(array: np.ndarray) -> Image.Image:
    """
    Convert numpy array to PIL Image.

    Args:
        array: Numpy array (H, W, C) or (H, W)

    Returns:
        PIL Image
    """
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
    # RGBのChannelsがないmaskデータ(H,W)の場合は、グレースケール(mode='L')
    if len(array.shape) == 2:
        return Image.fromarray(array, mode='L')
    else:
        return Image.fromarray(array, mode='RGB')


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """
    Convert PIL Image to numpy array.

    Args:
        image: PIL Image

    Returns:
        Numpy array
    """
    return np.array(image)


def create_empty_mask(width: int, height: int) -> np.ndarray:
    """
    Create an empty (all zeros) mask.

    Args:
        width: Mask width
        height: Mask height

    Returns:
        Empty mask array
    """
    return np.zeros((height, width), dtype=np.uint8)


def has_mask_content(mask: np.ndarray, threshold: float = 0.001) -> bool:
    """
    Check if mask contains any meaningful content.

    Args:
        mask: Binary mask array
        threshold: Minimum percentage of non-zero pixels (0-1)

    Returns:
        True if mask has content, False otherwise
    """
    if mask is None:
        return False

    non_zero_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    return (non_zero_pixels / total_pixels) > threshold
