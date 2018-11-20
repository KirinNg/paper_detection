import cv2
import numpy as np
import matplotlib.pyplot as plt

path = "demo.jpg"
kernel_size = (3, 3)
sigma = 10

img = cv2.imread(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),10)
ret,binary = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

_,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    rect = cv2.minAreaRect(c)
    # (2)计算最小面积矩形的坐标
    box = cv2.boxPoints(rect)
    # (3)坐标归一化为整型
    box = np.int0(box)
    # (4)绘制轮廓
draw_img = cv2.drawContours(img.copy(), [box], 0, (0, 255, 0), 3)

new_img = img[box[1][1]:box[3][1], box[1][0]:box[3][0]]

# plt.imshow(new_img)
# plt.show()
plt.imsave("out.png", new_img)