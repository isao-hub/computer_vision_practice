from pathlib import Path
import cv2 as cv
import numpy as np

# ディレクトリ設定
INPUT_DIR = Path("/Users/isaoishikawa/input_images")
OUTPUT_DIR = INPUT_DIR / "binary_canny_results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# パラメータ
BLUR_KERNEL     = 5
GLOBAL_THRESH   = 127
ADAPTIVE_BLOCK  = 11      # 奇数
ADAPTIVE_C      = 2

CANNY_LOW       = 100     # Cannyの下限閾値（小さくすると細い線も拾う）
CANNY_HIGH      = 200     # 上限閾値

# 出力フォルダ作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 対象画像取得
image_paths = []
for ext in IMAGE_EXTS:
    image_paths.extend(INPUT_DIR.glob(f"*{ext}"))
    image_paths.extend(INPUT_DIR.glob(f"*{ext.upper()}"))
image_paths = sorted(set(image_paths))

print(f"処理対象画像: {len(image_paths)} 枚")

# ─── バッチ処理 ───
for img_path in image_paths:
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        print("読み込み失敗")
        continue

    blur = cv.medianBlur(img, BLUR_KERNEL)
    
    # 0. Original → Canny edge detection
    original_canny = cv.Canny(blur, CANNY_LOW, CANNY_HIGH)

    # 1. Global Threshold(二値化)→ Canny edge detection
    ret, global_binary = cv.threshold(blur, GLOBAL_THRESH, 255, cv.THRESH_BINARY)
    global_canny = cv.Canny(global_binary, CANNY_LOW, CANNY_HIGH)

    # 2. Adaptive Mean(二値化) → Canny edge detection
    mean_binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
    mean_canny = cv.Canny(mean_binary, CANNY_LOW, CANNY_HIGH)

    # 3. Adaptive Gaussian(二値化) → Canny edge detection
    gauss_binary = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, ADAPTIVE_BLOCK, ADAPTIVE_C)
    gauss_canny = cv.Canny(gauss_binary, CANNY_LOW, CANNY_HIGH)

    # 保存
    stem = img_path.stem
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_original.jpg"), img)
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_original_canny.jpg"), global_canny)
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_global_canny.jpg"), global_canny)
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_mean_canny.jpg"),   mean_canny)
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_gaussian_canny.jpg"), gauss_canny)

    # 比較用（4つのCannyを横並びで1枚に）
    comparison = np.hstack([original_canny,global_canny, mean_canny, gauss_canny])
    cv.imwrite(str(OUTPUT_DIR / f"{stem}_comparison.jpg"), comparison)

print("全処理完了")