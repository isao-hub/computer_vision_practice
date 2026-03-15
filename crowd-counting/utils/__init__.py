"""Utility modules for hair/beard detection applications.

This package provides modular components for detection:
- ImageHandler: Image format conversion utilities
- RegionSelector: Rectangle extraction from Gradio ImageEditor
- BeardDetector: Unified detector with Grounded SAM, rule-based, and YOLO v8 backends
- DetectedRegion: Data class for detected regions
- visualize_single_hairs: Visualization utility
"""

from .image_handler import ImageHandler
from .region_selector import RegionSelector
from .beard_detector import DetectedRegion
from .single_hair_segmenter import visualize_single_hairs

from .beard_detector import (
    BeardDetector,
    GroundedSAMBackend,
    RuleBasedBackend,
    DetectedRegion,
    DetectionBackend,
)

# Single hair segmentation
from .single_hair_segmenter import (
    SingleHairSegmenter,
    SingleHairSegmentationPipeline,
    SeparationMethod,
    SegmentationConfig,
    visualize_single_hairs,
)

# Black/White hair detection
from .black_white_hair_detector import (
    BlackWhiteHairDetector,
    HairClassParams,
)

# Black/White hair detection - SAM2
from .black_white_hair_detector_sam2 import (
    BlackWhiteHairDetectorSAM2,
    SAM2_MODEL_CONFIGS,
)

# Hair thickness classification
from .thickness_classifier import (
    THICKNESS_CATEGORIES,
    ClassifiedHair,
    calculate_hair_width,
    classify_hair_thickness,
    visualize_classified_hairs,
)

# Morphology utilities
from .morphology_utils import (
    extract_skeleton,
    find_branch_endpoints,
    split_skeleton_at_branches,
    restore_segment_thickness,
    simple_connected_component_separation,
    preprocess_beard_mask,
    filter_by_shape,
    calculate_centroid,
)

__all__ = [
    # Image utilities
    'ImageHandler',
    
    # Region selection
    'RegionSelector',

    # Detection
    'BeardDetector',
    'GroundedSAMBackend',
    'RuleBasedBackend',
    'DetectedRegion',
    'DetectionBackend',

    # Single Hair Segmentation
    'SingleHairSegmenter',
    'SingleHairSegmentationPipeline',
    'SeparationMethod',
    'SegmentationConfig',
    'visualize_single_hairs',

    # Black/White Hair Detection
    'BlackWhiteHairDetector',
    'HairClassParams',

    # Black/White Hair Detection - SAM2
    'BlackWhiteHairDetectorSAM2',
    'SAM2_MODEL_CONFIGS',

    # Hair Thickness Classification
    'THICKNESS_CATEGORIES',
    'ClassifiedHair',
    'calculate_hair_width',
    'classify_hair_thickness',
    'visualize_classified_hairs',

    # Visualization
    'visualize_single_hairs',

    # Morphology Utilities
    'extract_skeleton',
    'find_branch_endpoints',
    'split_skeleton_at_branches',
    'restore_segment_thickness',
    'simple_connected_component_separation',
    'preprocess_beard_mask',
    'filter_by_shape',
    'calculate_centroid',
]

__version__ = '0.4.0'
