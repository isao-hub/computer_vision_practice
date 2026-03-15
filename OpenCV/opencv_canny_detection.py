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
img = cv.imread(image_path, 0)

if img is None: 
    print("エラー：画像が読み込めませんでした")
    exit()

edges = cv.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap="gray")
plt.title("Edge"), plt.xticks([]), plt.yticks([])
plt.show()