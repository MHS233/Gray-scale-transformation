import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from pylab import mpl

# mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 绘制直方图函数
def grayHist(img):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.xlabel("gray label")
    plt.ylabel("number of pixels")
    plt.axis([0, 255, 0, np.max(histogram)])
    plt.show()

#读图
img = cv.imread("image2.jpg", 0)
cv.imshow("YuanTu", img)

#直方图正规化
out = np.zeros(img.shape, np.uint8)
cv.normalize(img, out, 255, 0, cv.NORM_MINMAX, cv.CV_8U)

#对数变换
out2 = np.zeros(img.shape, np.uint8)
h, w = img.shape[:2]

for i in range(h):
    for j in range(w):
        pix = img[i][j]
        out2[i][j] = 40 * math.log(1 + pix)


out = np.around(out)
out = out.astype(np.uint8)

# 图像归一化
fi = img / 255.0
# 伽马变换
gamma = 0.4
out3 = np.power(fi, gamma)


# 创建CLAHE对象
img = cv.resize(img, None, fx=0.5, fy=0.5)
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# 限制对比度的自适应阈值均衡化
out4 = clahe.apply(img)
# 使用全局直方图均衡化
out5 = cv.equalizeHist(img)



cv.imshow("zhenggui", out)
cv.imshow("duishu", out2)
cv.imshow("gama", out3)
cv.imshow("QuanJuJunHeng", out4)
cv.imshow("ZiShiYingJunHeng", out5)

cv.waitKey()