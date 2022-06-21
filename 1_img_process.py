import cv2
import numpy as np

#이미지 불러오기
Number = ("images/semple_1.jpg")
img = cv2.imread(Number, cv2.IMREAD_COLOR)

#GRAY로 변환하기
copy_img=img.copy()
img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#Laplacian 적용
mask1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
laplacianed_img = cv2.filter2D(img2, -1, mask1)
canny = cv2.Canny(laplacianed_img, 100, 300)

#Gaussian 적용
blur = cv2.GaussianBlur(img2, (3,3), 0)
canny_blur = cv2.Canny(blur, 100, 200)


cv2.imwrite('img_processed.jpg', canny_blur)

"""
cv2.imshow("Original", img)
cv2.imshow("Gray Img", img2)
cv2.imshow("laplacianed_img ", laplacianed_img)
cv2.imshow("laplacianed Canny", canny)
cv2.imshow("GaussianBlur", canny_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""