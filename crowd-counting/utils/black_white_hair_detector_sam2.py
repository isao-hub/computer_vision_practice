"""Black and White Hair Detector using SAM2 Automatic Mask Generation.

This module provides detection for:
- Class 1: Black hair (dark hairs against lighter skin)
- Class 2: White hair (light/gray hairs against skin)

Each class has its own filter parameters for optimal detection.
Uses SAM2 (Segment Anything Model 2) with Hiera encoder for precise segmentation.

SAM2's Hiera (Hierarchical Vision Transformer) encoder features:
- Multi-scale hierarchical feature extraction (4 stages)
- Window-based attention with selective global attention
- Mask-to-Mask (M2M) refinement for improved mask quality

Supported models:
- sam2.1_hiera_tiny  (lightweight, faster)
- sam2.1_hiera_large (higher quality, slower)

Requirements:
    pip install sam2
    or: pip install git+https://github.com/facebookresearch/sam2.git
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List, Dict
import os
import time
from dataclasses import dataclass

from .beard_detector import DetectedRegion
from .black_white_hair_detector import HairClassParams

# Check SAM2 availability
SAM2_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    SAM2_AVAILABLE = True
except ImportError:
    pass


def get_device() -> str:
    """Detect the best available device for PyTorch."""
    if not TORCH_AVAILABLE:
        return "cpu"
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# SAM2 model configurations
# config paths are relative to the sam2 package's config directory (resolved by Hydra)
SAM2_MODEL_CONFIGS = {
    "sam2.1_hiera_tiny": {
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "label": "SAM2.1 Hiera Tiny (軽量・高速)",
    },
    "sam2.1_hiera_large": {
        "checkpoint": "sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "label": "SAM2.1 Hiera Large (高精度・低速)",
    },
}


class BlackWhiteHairDetectorSAM2:
    """
    Hair detector with separate black/white hair detection using SAM2.

    Uses SAM2 Automatic Mask Generation with Hiera encoder to detect
    individual hairs, then filters by brightness to classify as black or white hair.

    Key differences from SAM1 (BlackWhiteHairDetector):
    - Hiera encoder: hierarchical multi-scale features vs ViT single-scale
    - use_m2m: mask-to-mask refinement (SAM2-specific)
    - Different default thresholds (pred_iou_thresh=0.7, stability_score_thresh=0.8)
    - No MPS float64 patch needed (SAM2 uses float32 throughout)
    """

    def __init__(self):
        self._sam2_model = None
        self._sam2_amg: Optional['SAM2AutomaticMaskGenerator'] = None
        self._sam2_initialized = False
        self._current_model_name: Optional[str] = None
        # AMG parameter tracking for change detection
        self._current_points_per_side = 64
        self._current_pred_iou_thresh = 0.7
        self._current_stability_score_thresh = 0.8
        self._current_use_m2m = True
        self._current_multimask_output = True
        # Detailed timing for SAM2 internals
        self._timing_details: Dict[str, float] = {}
        self._timing_hooks_installed = False
        self.device = get_device()
        print(f"BlackWhiteHairDetectorSAM2 initialized with device: {self.device}")

    def _install_timing_hooks(self):
        """Install timing hooks on SAM2 model internals (image_encoder, prompt_encoder, mask_decoder)."""
        if self._timing_hooks_installed or self._sam2_model is None:
            return

        model = self._sam2_model

        # Find and wrap internal components
        components = {}
        for name in ['image_encoder', 'sam_prompt_encoder', 'sam_mask_decoder']:
            if hasattr(model, name):
                components[name] = getattr(model, name)

        if not components:
            print("[Timing] Could not find SAM2 model components, skipping hooks")
            return

        for comp_name, component in components.items():
            original_forward = component.forward

            def make_timed_forward(orig_fn, cname):
                def timed_forward(*args, **kwargs):
                    t0 = time.perf_counter()
                    result = orig_fn(*args, **kwargs)
                    elapsed = time.perf_counter() - t0
                    self._timing_details[cname] = self._timing_details.get(cname, 0.0) + elapsed
                    return result
                return timed_forward

            component.forward = make_timed_forward(original_forward, comp_name)

        self._timing_hooks_installed = True
        print(f"[Timing] Hooks installed on: {list(components.keys())}")

    def _reset_timing_details(self):
        """Reset per-inference timing accumulators."""
        self._timing_details = {}

    def _update_amg_params(
        self,
        points_per_side: int = 64,
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.8,
        use_m2m: bool = True,
        multimask_output: bool = True,
    ):
        """Update SAM2 AMG with new parameters if any changed."""
        if self._sam2_model is None or not self._sam2_initialized:
            return

        params_changed = (
            points_per_side != self._current_points_per_side
            or pred_iou_thresh != self._current_pred_iou_thresh
            or stability_score_thresh != self._current_stability_score_thresh
            or use_m2m != self._current_use_m2m
            or multimask_output != self._current_multimask_output
        )

        if params_changed:
            print(
                f"Updating SAM2 AMG: points_per_side={points_per_side}, "
                f"pred_iou_thresh={pred_iou_thresh}, "
                f"stability_score_thresh={stability_score_thresh}, "
                f"use_m2m={use_m2m}, multimask_output={multimask_output}"
            )
            self._sam2_amg = SAM2AutomaticMaskGenerator(
                self._sam2_model,
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                min_mask_region_area=0,
                use_m2m=use_m2m,
                multimask_output=multimask_output,
                output_mode="binary_mask",
                points_per_batch=256,
            )
            self._current_points_per_side = points_per_side
            self._current_pred_iou_thresh = pred_iou_thresh
            self._current_stability_score_thresh = stability_score_thresh
            self._current_use_m2m = use_m2m
            self._current_multimask_output = multimask_output

    def _init_sam2(self, model_name: str = "sam2.1_hiera_tiny") -> Tuple[bool, str]:
        """Initialize SAM2 model.

        Args:
            model_name: One of "sam2.1_hiera_tiny" or "sam2.1_hiera_large"
        """
        # If already initialized with the same model, skip
        if self._sam2_initialized and self._current_model_name == model_name:
            return True, f"SAM2 ({model_name}) already initialized"

        if not SAM2_AVAILABLE:
            return False, "SAM2 not available. Install with: pip install sam2"

        if model_name not in SAM2_MODEL_CONFIGS:
            return False, f"Unknown model: {model_name}. Available: {list(SAM2_MODEL_CONFIGS.keys())}"

        config = SAM2_MODEL_CONFIGS[model_name]
        checkpoint_filename = config["checkpoint"]
        model_cfg = config["config"]

        # Search for checkpoint file
        base_dir = os.path.dirname(__file__)
        checkpoint_paths = [
            os.path.join(base_dir, "..", "checkpoints", checkpoint_filename),
            os.path.join(base_dir, "checkpoints", checkpoint_filename),
            os.path.join("checkpoints", checkpoint_filename),
        ]

        sam2_path = None
        for path in checkpoint_paths:
            if os.path.exists(path):
                sam2_path = os.path.abspath(path)
                break

        if sam2_path is None:
            return False, (
                f"SAM2 checkpoint not found: {checkpoint_filename}\n"
                f"Searched: {[os.path.abspath(p) for p in checkpoint_paths]}\n"
                f"Download from: https://github.com/facebookresearch/sam2"
            )

        try:
            print(f"Loading SAM2 ({model_name}) from: {sam2_path}")

            # Release previous model if switching
            if self._sam2_model is not None:
                del self._sam2_model
                del self._sam2_amg
                self._sam2_model = None
                self._sam2_amg = None
                if TORCH_AVAILABLE:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            sam2_model = build_sam2(
                model_cfg,
                sam2_path,
                device=self.device,
            )

            self._sam2_model = sam2_model
            # min_mask_region_area=0: skip postprocess_small_regions (CPU bottleneck)
            # Our own _filter_masks handles min_area filtering instead
            # points_per_batch=256: increase GPU parallelism (default=64)
            self._sam2_amg = SAM2AutomaticMaskGenerator(
                sam2_model,
                points_per_side=self._current_points_per_side,
                pred_iou_thresh=self._current_pred_iou_thresh,
                stability_score_thresh=self._current_stability_score_thresh,
                min_mask_region_area=0,
                use_m2m=self._current_use_m2m,
                multimask_output=self._current_multimask_output,
                output_mode="binary_mask",
                points_per_batch=256,
            )
            self._sam2_initialized = True
            self._current_model_name = model_name
            self._timing_hooks_installed = False
            self._install_timing_hooks()
            print(f"SAM2 ({model_name}) initialized on {self.device}")
            return True, f"SAM2 ({model_name}) initialized on {self.device}"
        except Exception as e:
            return False, f"SAM2 initialization error: {e}"

    def detect_with_class(
        self,
        image_rgb: np.ndarray,
        region_box: Tuple[int, int, int, int],
        hair_class: str,  # 'black' or 'white'
        params: HairClassParams,
        points_per_side: int = 64,
        use_tiling: bool = False,
        tile_size: int = 400,
        tile_overlap: int = 50,
        overlap_threshold: float = 0.5,
        model_name: str = "sam2.1_hiera_tiny",
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.8,
        use_m2m: bool = True,
        multimask_output: bool = True,
    ) -> Tuple[List[DetectedRegion], List[np.ndarray], Dict]:
        """
        Detect hairs of a specific color class using SAM2.

        Args:
            image_rgb: Full RGB image
            region_box: (x1, y1, x2, y2) ROI
            hair_class: 'black' or 'white'
            params: HairClassParams with filter settings
            points_per_side: SAM2 sampling density
            use_tiling: Enable tile-based processing
            tile_size: Size of each tile
            tile_overlap: Overlap between tiles
            overlap_threshold: Duplicate removal threshold
            model_name: SAM2 model to use
            pred_iou_thresh: SAM2 IoU prediction threshold
            stability_score_thresh: SAM2 stability score threshold
            use_m2m: Enable mask-to-mask refinement (SAM2-specific)
            multimask_output: Enable multi-mask output

        Returns:
            Tuple of (filtered detections, all masks, stats dict)
        """
        empty_stats = {
            'total': 0, 'filtered_area_small': 0, 'filtered_area_large': 0,
            'filtered_aspect': 0, 'filtered_brightness': 0, 'passed': 0,
            'hair_class': hair_class, 'model': model_name,
        }

        if not self._sam2_initialized or self._current_model_name != model_name:
            success, msg = self._init_sam2(model_name)
            if not success:
                print(f"SAM2 not available: {msg}")
                return [], [], empty_stats

        self._update_amg_params(
            points_per_side, pred_iou_thresh,
            stability_score_thresh, use_m2m, multimask_output,
        )

        x1, y1, x2, y2 = region_box
        h, w = image_rgb.shape[:2]

        # Crop ROI
        roi = image_rgb[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        # Tiled processing
        if use_tiling:
            tiles = self._calculate_tiles(roi_w, roi_h, tile_size, tile_overlap)
            print(f"Tile-based processing: {len(tiles)} tiles")

            all_results = []
            all_masks_unfiltered = []
            stats = {
                'total': 0, 'filtered_area_small': 0, 'filtered_area_large': 0,
                'filtered_no_contour': 0, 'filtered_zero_dim': 0,
                'filtered_aspect': 0, 'filtered_brightness': 0, 'passed': 0,
                'tiles': len(tiles), 'hair_class': hair_class, 'model': model_name,
            }

            t_tile_amg_total = 0.0
            t_tile_store_total = 0.0
            t_tile_filter_total = 0.0
            self._reset_timing_details()

            for idx, (tx1, ty1, tx2, ty2) in enumerate(tiles):
                tile_rgb = roi[ty1:ty2, tx1:tx2]
                print(f"  Processing tile {idx+1}/{len(tiles)}")

                t_tile_amg_s = time.perf_counter()
                if self.device == "cuda" and TORCH_AVAILABLE:
                    import torch
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        tile_masks = self._sam2_amg.generate(tile_rgb)
                else:
                    tile_masks = self._sam2_amg.generate(tile_rgb)
                t_tile_amg_total += time.perf_counter() - t_tile_amg_s
                stats['total'] += len(tile_masks)

                # Store all masks
                t_tile_store_s = time.perf_counter()
                tile_h, tile_w = tile_rgb.shape[:2]
                abs_x = x1 + tx1
                abs_y = y1 + ty1
                for mask_data in tile_masks:
                    full_mask = np.zeros((h, w), dtype=np.uint8)
                    mask = mask_data['segmentation'].astype(np.uint8) * 255
                    full_mask[abs_y:abs_y+tile_h, abs_x:abs_x+tile_w] = mask
                    all_masks_unfiltered.append(full_mask)
                t_tile_store_total += time.perf_counter() - t_tile_store_s

                t_tile_filter_s = time.perf_counter()
                tile_results, tile_stats = self._filter_tile_masks(
                    tile_masks, tile_rgb, (h, w),
                    (tx1, ty1), (x1, y1), params, hair_class
                )
                t_tile_filter_total += time.perf_counter() - t_tile_filter_s
                all_results.extend(tile_results)

                # Aggregate stats
                for key in ['filtered_area_small', 'filtered_area_large', 'filtered_no_contour', 'filtered_zero_dim', 'filtered_aspect', 'filtered_brightness']:
                    stats[key] += tile_stats.get(key, 0)

            # Remove duplicates
            t_dedup_start = time.perf_counter()
            all_results = self._remove_duplicates(all_results, overlap_threshold)
            t_dedup_end = time.perf_counter()
            stats['passed'] = len(all_results)

            timing = {
                'sam2_amg_generate': t_tile_amg_total,
                'mask_store': t_tile_store_total,
                'mask_filter': t_tile_filter_total,
                'duplicate_removal': t_dedup_end - t_dedup_start,
            }
            timing['detector_total'] = sum(timing.values())

            # Add SAM2 internal breakdown
            td = self._timing_details
            t_img_enc = td.get('image_encoder', 0)
            t_prompt_enc = td.get('sam_prompt_encoder', 0)
            t_mask_dec = td.get('sam_mask_decoder', 0)
            t_amg_other = t_tile_amg_total - t_img_enc - t_prompt_enc - t_mask_dec
            timing['image_encoder'] = t_img_enc
            timing['prompt_encoder'] = t_prompt_enc
            timing['mask_decoder'] = t_mask_dec
            timing['amg_other'] = t_amg_other
            stats['timing'] = timing

            self._print_timing_table(timing, tiled_count=len(tiles))

            return all_results, all_masks_unfiltered, stats

        # Non-tiled processing
        print("Running SAM2 Automatic Mask Generation...")
        self._reset_timing_details()
        t_amg_start = time.perf_counter()
        if self.device == "cuda" and TORCH_AVAILABLE:
            import torch
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = self._sam2_amg.generate(roi)
        else:
            masks = self._sam2_amg.generate(roi)
        t_amg_end = time.perf_counter()
        print(f"SAM2 AMG generated {len(masks)} masks")

        # Store all masks
        t_store_start = time.perf_counter()
        all_masks_unfiltered = []
        for mask_data in masks:
            full_mask = np.zeros((h, w), dtype=np.uint8)
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            full_mask[y1:y2, x1:x2] = mask
            all_masks_unfiltered.append(full_mask)
        t_store_end = time.perf_counter()

        # Filter masks
        t_filter_start = time.perf_counter()
        results, stats = self._filter_masks(
            masks, roi, (h, w), region_box, params, hair_class
        )
        stats['model'] = model_name
        t_filter_end = time.perf_counter()

        # Remove duplicates
        t_dedup_start = time.perf_counter()
        results = self._remove_duplicates(results, overlap_threshold)
        stats['passed'] = len(results)
        t_dedup_end = time.perf_counter()

        # Timing report
        t_amg_total = t_amg_end - t_amg_start
        timing = {
            'sam2_amg_generate': t_amg_total,
            'mask_store': t_store_end - t_store_start,
            'mask_filter': t_filter_end - t_filter_start,
            'duplicate_removal': t_dedup_end - t_dedup_start,
        }
        timing['detector_total'] = sum(timing.values())

        # Add SAM2 internal breakdown
        td = self._timing_details
        t_img_enc = td.get('image_encoder', 0)
        t_prompt_enc = td.get('sam_prompt_encoder', 0)
        t_mask_dec = td.get('sam_mask_decoder', 0)
        t_amg_other = t_amg_total - t_img_enc - t_prompt_enc - t_mask_dec
        timing['image_encoder'] = t_img_enc
        timing['prompt_encoder'] = t_prompt_enc
        timing['mask_decoder'] = t_mask_dec
        timing['amg_other'] = t_amg_other
        stats['timing'] = timing

        self._print_timing_table(timing)

        return results, all_masks_unfiltered, stats

    def _print_timing_table(self, timing: Dict[str, float], tiled_count: int = 0):
        """Print a formatted timing breakdown table."""
        total = timing['detector_total']
        t_amg = timing['sam2_amg_generate']

        header = f"SAM2 Detector Timing (Tiled x{tiled_count})" if tiled_count else "SAM2 Detector Timing Breakdown"
        print(f"\n┌───────────────────────────────────────────────────┐")
        print(f"│ {header:<49} │")
        print("├──────────────────────────────┬──────────┬─────────┤")
        print("│ Step                         │ Time (s) │    %    │")
        print("├──────────────────────────────┼──────────┼─────────┤")

        def row(label, t, ref=total):
            pct = (t / ref * 100) if ref > 0 else 0
            print(f"│ {label:<28} │ {t:>8.3f} │ {pct:>6.1f}% │")

        row("SAM2 AMG Generate", t_amg)

        # AMG internal breakdown
        if timing.get('image_encoder', 0) > 0 or timing.get('prompt_encoder', 0) > 0:
            row("  Image Encoder (Hiera)", timing['image_encoder'])
            row("  Prompt Encoder", timing['prompt_encoder'])
            row("  Mask Decoder", timing['mask_decoder'])
            row("  AMG Other (NMS etc.)", timing['amg_other'])

        row("Mask Store (full-size)", timing['mask_store'])
        row("Mask Filter", timing['mask_filter'])
        row("Duplicate Removal", timing['duplicate_removal'])
        print("├──────────────────────────────┼──────────┼─────────┤")
        print(f"│ {'Detector Total':<28} │ {total:>8.3f} │ 100.0%  │")
        print("└──────────────────────────────┴──────────┴─────────┘\n")

    def _filter_masks(
        self,
        masks: List[Dict],
        roi: np.ndarray,
        full_shape: Tuple[int, int],
        region_box: Tuple[int, int, int, int],
        params: HairClassParams,
        hair_class: str
    ) -> Tuple[List[DetectedRegion], Dict]:
        """Filter SAM2 masks based on hair class parameters."""
        x1, y1, x2, y2 = region_box
        h, w = full_shape

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(roi_gray)

        stats = {
            'total': len(masks),
            'filtered_area_small': 0,
            'filtered_area_large': 0,
            'filtered_no_contour': 0,
            'filtered_zero_dim': 0,
            'filtered_aspect': 0,
            'filtered_brightness': 0,
            'passed': 0,
            'hair_class': hair_class,
            'mean_brightness': float(mean_brightness)
        }

        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            area = mask_data['area']

            # Area filter
            if area < params.min_area:
                stats['filtered_area_small'] += 1
                continue
            if area > params.max_area:
                stats['filtered_area_large'] += 1
                continue

            # Aspect ratio filter
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                stats['filtered_no_contour'] += 1
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            if w_rect == 0:
                w_rect = 1.0
            if h_rect == 0:
                h_rect = 1.0
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect < params.min_aspect:
                stats['filtered_aspect'] += 1
                continue

            # Brightness filter (class-dependent)
            mask_brightness = cv2.mean(roi_gray, mask=mask)[0]

            if hair_class == 'black':
                if mask_brightness > mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue
            else:  # white
                if mask_brightness < mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue

            # Create full-size mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = mask

            # Apply dilation if enabled
            if params.dilation_kernel_size > 0:
                ksize = params.dilation_kernel_size
                if ksize % 2 == 0:
                    ksize += 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                full_mask = cv2.dilate(
                    full_mask, kernel, iterations=params.dilation_iterations
                )

                new_area = cv2.countNonZero(full_mask)
                if new_area > params.max_area:
                    stats['filtered_area_large'] += 1
                    continue
                area = new_area

            # Calculate centroid
            M = cv2.moments(full_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2

            stats['passed'] += 1

            results.append(DetectedRegion(
                mask=full_mask,
                area=area,
                centroid=(cx, cy),
                confidence=mask_data.get('stability_score', 0.5),
                source=f'sam2_amg_{hair_class}',
                phrase=f"{hair_class}_hair_{i+1}"
            ))

        print(f"Filter stats ({hair_class}): {stats}")
        return results, stats

    def _filter_tile_masks(
        self,
        masks: List[Dict],
        tile_rgb: np.ndarray,
        full_shape: Tuple[int, int],
        tile_offset: Tuple[int, int],
        region_offset: Tuple[int, int],
        params: HairClassParams,
        hair_class: str
    ) -> Tuple[List[DetectedRegion], Dict]:
        """Filter masks from a single tile."""
        h, w = full_shape
        tx, ty = tile_offset
        rx, ry = region_offset
        tile_h, tile_w = tile_rgb.shape[:2]

        tile_gray = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(tile_gray)

        stats = {
            'filtered_area_small': 0, 'filtered_area_large': 0,
            'filtered_no_contour': 0, 'filtered_zero_dim': 0,
            'filtered_aspect': 0, 'filtered_brightness': 0
        }

        results = []
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'].astype(np.uint8) * 255
            area = mask_data['area']

            if area < params.min_area:
                stats['filtered_area_small'] += 1
                continue
            if area > params.max_area:
                stats['filtered_area_large'] += 1
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                stats['filtered_no_contour'] += 1
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            if w_rect == 0:
                w_rect = 1.0
            if h_rect == 0:
                h_rect = 1.0
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect < params.min_aspect:
                stats['filtered_aspect'] += 1
                continue

            mask_brightness = cv2.mean(tile_gray, mask=mask)[0]

            if hair_class == 'black':
                if mask_brightness > mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue
            else:
                if mask_brightness < mean_brightness * params.brightness_threshold:
                    stats['filtered_brightness'] += 1
                    continue

            # Create full-size mask
            full_mask = np.zeros((h, w), dtype=np.uint8)
            abs_x = rx + tx
            abs_y = ry + ty
            full_mask[abs_y:abs_y+tile_h, abs_x:abs_x+tile_w] = mask

            # Apply dilation if enabled
            if params.dilation_kernel_size > 0:
                ksize = params.dilation_kernel_size
                if ksize % 2 == 0:
                    ksize += 1
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (ksize, ksize)
                )
                full_mask = cv2.dilate(
                    full_mask, kernel, iterations=params.dilation_iterations
                )

                new_area = cv2.countNonZero(full_mask)
                if new_area > params.max_area:
                    stats['filtered_area_large'] += 1
                    continue
                area = new_area

            # Calculate centroid
            M = cv2.moments(full_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx = abs_x + tile_w // 2
                cy = abs_y + tile_h // 2

            results.append(DetectedRegion(
                mask=full_mask,
                area=area,
                centroid=(cx, cy),
                confidence=mask_data.get('stability_score', 0.5),
                source=f'sam2_amg_{hair_class}_tiled',
                phrase=f"{hair_class}_hair_{i+1}"
            ))

        return results, stats

    def _calculate_tiles(
        self, region_width: int, region_height: int,
        tile_size: int = 400, overlap: int = 50
    ) -> List[Tuple[int, int, int, int]]:
        """Calculate tile positions."""
        tiles = []

        if region_width <= tile_size and region_height <= tile_size:
            return [(0, 0, region_width, region_height)]

        step = max(tile_size - overlap, 1)

        y = 0
        while y < region_height:
            x = 0
            while x < region_width:
                x1, y1 = x, y
                x2 = min(x + tile_size, region_width)
                y2 = min(y + tile_size, region_height)

                if x2 - x1 >= overlap and y2 - y1 >= overlap:
                    tiles.append((x1, y1, x2, y2))

                x += step
                if x2 >= region_width:
                    break

            y += step
            if y2 >= region_height:
                break

        return tiles

    def _remove_duplicates(
        self, detections: List[DetectedRegion], overlap_threshold: float = 0.5
    ) -> List[DetectedRegion]:
        """Remove duplicate detections."""
        if len(detections) <= 1:
            return detections

        sorted_dets = sorted(detections, key=lambda x: x.confidence, reverse=True)
        unique = []

        for det in sorted_dets:
            is_duplicate = False
            for existing in unique:
                intersection = cv2.bitwise_and(det.mask, existing.mask)
                intersection_area = cv2.countNonZero(intersection)
                min_area = min(det.area, existing.area)
                if min_area > 0 and intersection_area / min_area > overlap_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(det)

        return unique

    def detect_with_class_and_mask(
        self,
        image_rgb: np.ndarray,
        region_mask: np.ndarray,
        hair_class: str,
        params: HairClassParams,
        points_per_side: int = 64,
        use_tiling: bool = False,
        tile_size: int = 400,
        tile_overlap: int = 50,
        overlap_threshold: float = 0.5,
        model_name: str = "sam2.1_hiera_tiny",
        pred_iou_thresh: float = 0.7,
        stability_score_thresh: float = 0.8,
        use_m2m: bool = True,
        multimask_output: bool = True,
    ) -> Tuple[List[DetectedRegion], List[np.ndarray], Dict]:
        """
        Detect hairs within a freeform mask region using SAM2.

        Args:
            image_rgb: Full RGB image
            region_mask: Binary mask (255 = detection region)
            hair_class: 'black' or 'white'
            params: HairClassParams with filter settings
            points_per_side: SAM2 sampling density
            use_tiling: Enable tile-based processing
            tile_size: Size of each tile
            tile_overlap: Overlap between tiles
            overlap_threshold: Duplicate removal threshold
            model_name: SAM2 model to use
            pred_iou_thresh: SAM2 IoU prediction threshold
            stability_score_thresh: SAM2 stability score threshold
            use_m2m: Enable mask-to-mask refinement (SAM2-specific)
            multimask_output: Enable multi-mask output

        Returns:
            Tuple of (filtered detections, all masks, stats dict)
        """
        # Ensure mask is binary
        mask_binary = (region_mask > 128).astype(np.uint8) * 255

        # Get bounding box from mask
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            empty_stats = {
                'total': 0, 'filtered_area_small': 0, 'filtered_area_large': 0,
                'filtered_aspect': 0, 'filtered_brightness': 0, 'passed': 0,
                'filtered_outside_mask': 0, 'hair_class': hair_class,
                'selection_mode': 'freeform', 'model': model_name,
            }
            return [], [], empty_stats

        # Compute bounding box that encompasses all contours
        x_min, y_min = image_rgb.shape[1], image_rgb.shape[0]
        x_max, y_max = 0, 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        region_box = (x_min, y_min, x_max, y_max)

        # Run detection with bounding box
        _, all_masks_raw, base_stats = self.detect_with_class(
            image_rgb, region_box, hair_class, params,
            points_per_side=points_per_side,
            use_tiling=use_tiling,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            overlap_threshold=1.0,  # Skip duplicate removal here
            model_name=model_name,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            use_m2m=use_m2m,
            multimask_output=multimask_output,
        )

        # Re-filter using freeform mask for accurate mean_brightness calculation
        image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        mask_pixels = image_gray[mask_binary > 0]
        if len(mask_pixels) > 0:
            mean_brightness = np.mean(mask_pixels)
        else:
            mean_brightness = np.mean(image_gray[y_min:y_max, x_min:x_max])

        h, w = image_rgb.shape[:2]
        filtered_detections = []
        filtered_outside_mask = 0
        filtered_area_small = 0
        filtered_area_large = 0
        filtered_aspect = 0
        filtered_brightness = 0

        for full_mask in all_masks_raw:
            # Clip mask to freeform region
            masked = cv2.bitwise_and(full_mask, mask_binary)
            area = cv2.countNonZero(masked)

            if area == 0:
                filtered_outside_mask += 1
                continue

            # Area filter
            if area < params.min_area:
                filtered_area_small += 1
                continue
            if area > params.max_area:
                filtered_area_large += 1
                continue

            # Aspect ratio filter
            contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue

            rect = cv2.minAreaRect(contours[0])
            w_rect, h_rect = rect[1]
            if w_rect == 0:
                w_rect = 1.0
            if h_rect == 0:
                h_rect = 1.0
            aspect = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect < params.min_aspect:
                filtered_aspect += 1
                continue

            # Brightness filter using freeform-aware mean_brightness
            mask_brightness = cv2.mean(image_gray, mask=masked)[0]

            if hair_class == 'black':
                if mask_brightness > mean_brightness * params.brightness_threshold:
                    filtered_brightness += 1
                    continue
            else:  # white
                if mask_brightness < mean_brightness * params.brightness_threshold:
                    filtered_brightness += 1
                    continue

            # Apply dilation if enabled
            final_mask = masked
            if params.dilation_kernel_size > 0:
                ksize = params.dilation_kernel_size
                if ksize % 2 == 0:
                    ksize += 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                final_mask = cv2.dilate(masked, kernel, iterations=params.dilation_iterations)
                # Re-clip to freeform mask after dilation
                final_mask = cv2.bitwise_and(final_mask, mask_binary)
                area = cv2.countNonZero(final_mask)
                if area > params.max_area:
                    filtered_area_large += 1
                    continue

            # Calculate centroid
            M = cv2.moments(final_mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2

            filtered_detections.append(DetectedRegion(
                mask=final_mask,
                area=area,
                centroid=(cx, cy),
                confidence=0.5,
                source=f'sam2_amg_{hair_class}_freeform',
                phrase=f"{hair_class}_hair"
            ))

        # Remove duplicates
        filtered_detections = self._remove_duplicates(filtered_detections, overlap_threshold)

        stats = {
            'total': len(all_masks_raw),
            'filtered_area_small': filtered_area_small,
            'filtered_area_large': filtered_area_large,
            'filtered_aspect': filtered_aspect,
            'filtered_brightness': filtered_brightness,
            'filtered_outside_mask': filtered_outside_mask,
            'passed': len(filtered_detections),
            'hair_class': hair_class,
            'selection_mode': 'freeform',
            'mean_brightness': float(mean_brightness),
            'model': model_name,
        }

        return filtered_detections, all_masks_raw, stats
