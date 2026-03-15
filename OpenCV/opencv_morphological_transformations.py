import os
import cv2 as cv # type: ignore
import numpy as np # type: ignore
from matplotlib import pyplot as plt # type: ignore

# 画像を格納している画像フォルダの指定
IMAGE_DIR = "/Users/isaoishikawa/input_images"

# 読込みたいファイルを指定
filename = "beard1.jpeg"
# パス結合
image_path = os.path.join(IMAGE_DIR, filename)

# 画像読込み
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None: 
    print("エラー：画像が読み込めませんでした")
    exit()

kernel = np.ones((5,5), np.unit8)

# 1. Erosion
erosion = cv.erode(img, kernel, iterations=1)
# 2. Dilation
dilation = cv.dilate(img, kernel, iterations=1)
# 3. Opening
opening = cv.morphologyEx(img, cv.MORPH_OPEN,kernel)
# 4. Closing
cloasing = cv.morphologyEx(img, cv.MORPH_CLOSE,kernel)
# 5. Morphological Gradient
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT,kernel)
# 6. Top Hat
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT,kernel)
# 7. Black Hat
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT,kernel)
