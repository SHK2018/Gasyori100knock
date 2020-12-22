# -*- coding: utf-8 -*-
import cv2

img = cv2.imread("Jeanne.jpg")
cv2.imshow("Jeanne", img)
cv2.waitKey(0)
cv2.destroyAllWindows()