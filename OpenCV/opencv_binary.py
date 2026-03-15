import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 画像を格納している画像フォルダの指定
IMAGE_DIR = "/Users/isaoishikawa/input_images"

# 読込みたいファイルを指定
filename = "beard1.jpeg"

# パス結合
image_path = os.path.join(IMAGE_DIR, filename)

# 画像読込み (OpenCVでは二値画像に対する処理が基本)
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

if img is None: 
    print("エラー：画像が読み込めませんでした")
    exit()

# OpenCVで行える二値化の方法
## Global Thresholding（大域的二値化）
### 閾値ベース(0~255に該当する黒白の256の値の中間である127で二値化)
ret, thresh1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY) 
ret,thresh2 = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)

# 可視化
titles = ["Original", 'Binary', 
          'Binary_INV', 'Trunce', 
          'ToZero', 'ToZero_INV',]
images = [img, thresh1, 
          thresh2, thresh3, 
          thresh4, thresh5,]
for i in range(6):
    plt.subplot(2,3, i+1),plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
### 範囲ベース 

## Adaptive Thresholding（適応的二値化）
