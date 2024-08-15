import cv2
from matplotlib import pyplot as plt

# อ่านภาพจากไฟล์
image = cv2.imread('example.jpg')

# แสดงภาพ
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
