from pathlib import Path
import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# ディレクトリ設定
INPUT_DIR = Path("/Users/isaoishikawa/input_images")
OUTPUT_DIR = INPUT_DIR / "binary_results"
IMAGE_EXTS = {".jpg", "jpeg",".png", "bmp",".tiff", "tif",}

# 二値化パラメータの初期化
BLUR_KERNEL = 5
GLOBAL_THRESH = 127
ADAPTIVE_BLOCK = 11
ADAPTIVE_C = 2

# 出力先のフォルダがない場合に作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 対象画像のパスをすべて取得
image_paths = []
for ext in IMAGE_EXTS:
    image_paths.extend(INPUT_DIR.glob(f"*{ext}"))
    image_paths.extend(INPUT_DIR.glob(f"*{ext.upper}"))

image_paths = sorted(set(image_paths)) # 重複除去 & ソート

print(f"処理対象画像: {len(image_paths)}枚")

# 画像読込み (OpenCVでは二値画像に対する処理が基本)
for img_path in image_paths:
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None: 
        print("エラー：画像が読み込めませんでした")
        continue
    # 前処理（ノイズの軽減）
    img_blur = cv.medianBlur(img, 5)
    
    # Simple Thresholding（二値化）
    ret, th1 = cv.threshold(img_blur, GLOBAL_THRESH, 255, cv.THRESH_BINARY)
    
    ## Adaptive Thresholding（適応的二値化）：Mean
    th2 = cv.adaptiveThreshold(img_blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,ADAPTIVE_BLOCK,ADAPTIVE_C)
    
    ## Adaptive Thresholding（適応的二値化）: Gaussian
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,ADAPTIVE_BLOCK,ADAPTIVE_C)

    # 保存
    stem = img_path.stem #拡張子なしのファイル名

    cv.imwrite(str(OUTPUT_DIR / f"{stem}_original.jpg"), img)
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_global.jpg"), th1)
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_mean.jpg"), th2)
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_gaussian.jpg"), th3)

print("全画像の二値化処理が完了")