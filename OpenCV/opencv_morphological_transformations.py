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

# 画像読込み (OpenCVではグレースケールに対する処理)
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None: 
    print("エラー：画像が読み込めませんでした")
    exit()

kernel = np.ones((5,5), np.uint8)

# 1. Erosion
erosion = cv.erode(img, kernel, iterations=1)
# 2. Dilation
dilation = cv.dilate(img, kernel, iterations=1)
# 3. Opening
opening = cv.morphologyEx(img, cv.MORPH_OPEN,kernel)
# 4. Closing
closing = cv.morphologyEx(img, cv.MORPH_CLOSE,kernel)
# 5. Morphological Gradient
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT,kernel)
# 6. Top Hat
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT,kernel)
# 7. Black Hat
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT,kernel)

# 可視化
titles = ["Original", 'Erosion', 
          'Dilation', 'Opening', 
          'Closing', 'Gradient',
          'Top Hat', 'Black Hat']
images = [img, erosion, 
          dilation, opening, 
          closing, gradient, 
          tophat, blackhat]

plt.figure(figsize=(12,8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([]) #目盛り値の設定オフ
plt.tight_layout()
plt.show()