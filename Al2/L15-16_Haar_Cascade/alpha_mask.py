import cv2
import numpy as np

img = cv2.imread("Al2/L15-16 Haar_Cascade/mustache.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, alpha = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

b, g, r = cv2.split(img)
rgba = cv2.merge([b, g, r, alpha])
cv2.imwrite("mustache_alpha_channel.png", rgba)
