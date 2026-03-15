from pathlib import Path
import os
import cv2 as cv
import numpy as np

# ディレクトリ設定
INPUT_DIR = Path("/Users/isaoishikawa/input_images")
OUTPUT_DIR = INPUT_DIR / "edge-detection_results"
IMAGE_EXTS = {".jpg", "jpeg",".png", "bmp",".tiff", "tif",}

# 出力先のフォルダがない場合に作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 対象画像をすべて取得
image_paths = []
for ext in IMAGE_EXTS:
    image_paths.extend(INPUT_DIR.glob(f"*{ext}"))
    image_paths.extend(INPUT_DIR.glob(f"*{ext.upper()}")) #.JPEGも考慮

# 重複除去とソート
image_paths = sorted(set(image_paths))

# Canny Detection処理
for img_path in image_paths:
    img = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"エラー：画像が読み込めませんでした:{img_path.name}")
        continue
    edges = cv.Canny(img, 100, 200)
    # 保存先のパス
    output_path = OUTPUT_DIR / img_path.name

    #結果を保存
    combined = np.vstack([img, edges])
    cv.imwrite(str(output_path), combined)

    print(f"処理完了")
print(f"全処理完了")
 
