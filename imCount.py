import cv2
import numpy as np
img = cv2.imread('test.png',cv2.IMREAD_GRAYSCALE)
template=cv2.imread('template.png',cv2.IMREAD_GRAYSCALE)

# 二值化
ret, dst = cv2.threshold(img, 200,255,cv2.THRESH_BINARY_INV)

# 十字形卷积核很关键MORPH_CROSS
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
# 腐蚀操作
dst = cv2.erode(dst, element,iterations=3)
# 膨胀操作
dst = cv2.dilate(dst, element,iterations=5)



#方案1 联通区域检测

tmp=dst.copy()
# 取反 因为联通检测测的是白色的联通区域
for item in np.nditer(tmp, op_flags=['readwrite']):
    item[...] = 255-item

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(tmp, connectivity=8)
print(centroids)
color2=cv2.cvtColor(tmp,cv2.COLOR_GRAY2RGB)
for i in centroids:
    x,y=i
    cv2.circle(color2,(int(x),int(y)),4,(0,0,255),-1)
cv2.imshow('1',color2)
cv2.waitKey()
# 方案2拐角检测
gray=np.float32(dst)

corners=cv2.goodFeaturesToTrack(gray,100,0.5,10)
corners=np.int0(corners)
color=cv2.cvtColor(dst,cv2.COLOR_GRAY2RGB)
for i in corners:
    x,y=i.ravel()
    cv2.circle(color,(x,y),4,(0,0,255),-1)
cv2.imshow('1',color)
cv2.waitKey()