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
### 閾値ベース(0~255に該当する黒白の256の値の中間である127で二値化する場合)
"""Argument: \
    src: 入力画像 ndaaray,\
    threshold: 閾値 int \
    maxValue: 前景の値 int \
    thresholdType: 二値化の方法 int"""

ret, binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY) 
ret,binary_inv = cv.threshold(img,127,255,cv.THRESH_BINARY_INV)
ret,trunc = cv.threshold(img,127,255,cv.THRESH_TRUNC)
ret,tozero = cv.threshold(img,127,255,cv.THRESH_TOZERO)
ret,tozero_inv = cv.threshold(img,127,255,cv.THRESH_TOZERO_INV)
ret,binary_triangle = cv.threshold(img,127,255,cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

# 可視化
fig1 = plt.figure(figsize=(12,6))
titles = ["Original", 'Binary', 
          'Binary_INV', 'Trunce', 
          'ToZero', 'ToZero_INV',
          'Triangle']
images = [img, binary, 
          binary_inv, trunc, 
          tozero, tozero_inv,
          binary_triangle]
for i in range(7):
    plt.subplot(2,4, i+1),plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()

### 範囲ベース 割愛

## Adaptive Thresholding（適応的二値化）
img = cv.medianBlur(img, 5)
ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,11,2)

# 可視化
fig2 = plt.figure(figsize=(10,8))
titles_2 = ["Original", 'Global', 
          'Adaptive Mean', 'Adaptive Gaussian',]
images_2 = [img, th1, 
          th2, th3,]

for i in range(4):
    plt.subplot(2,2, i+1),plt.imshow(images_2[i], 'gray', vmin=0, vmax=255)
    plt.title(titles_2[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()