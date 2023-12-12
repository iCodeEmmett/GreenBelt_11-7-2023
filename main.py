import numpy as np
import math
import cv2 as cv
import sys

cap = cv.VideoCapture(0)
_, img = cap.read()
if img is None:
		print("Failed to capture image from camera. Check if your webcam device is working.")
		cap.release()  # Release the camera resource
		cv.destroyAllWindows()  # Close the OpenCV window
		sys.exit()  # Exit the program to prevent further errors

cv.rectangle(img, (300, 300), (100, 100), (0, 255, 0), 0)
crop_img = img[100:300, 100:300]
drawing = np.zeros(crop_img.shape, np.uint8)  # Initialize drawing variable
grey = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
value = (35, 35)
blurred = cv.GaussianBlur(grey, value, 0)
_, thresholded = cv.threshold(blurred, 127, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

contours = None
try:
		# Depending on the OpenCV version, findContours will return different values
		# Use this if your OpenCV returns two values: contours, hierarchy
		contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
		# Use this if your OpenCV returns three values: image, contours, hierarchy
		# image, contours, hierarchy = cv.findContours(thresholded.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
except cv.error as e:
		print(f"Error during findContours: {e}")

if contours and len(contours) > 0:
		count1 = max(contours, key=lambda x: cv.contourArea(x))
		x, y, w, h = cv.boundingRect(count1)
		cv.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 0)
		hull = cv.convexHull(count1)
		drawing = np.zeros(crop_img.shape, np.uint8)
		cv.drawContours(drawing, [count1], 0, (0, 255, 0), 0)
		cv.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
		hull = cv.convexHull(count1, returnPoints=False)
		defects = cv.convexityDefects(count1, hull)
		if defects is not None:
						count_defects = 0

						for i in range(defects.shape[0]):
										s, e, f, d = defects[i, 0]
										start = tuple(count1[s][0])
										end = tuple(count1[e][0])
										far = tuple(count1[f][0])
										a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
										b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
										c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
										angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

										if angle <= 90:
														count_defects += 1
														cv.circle(crop_img, far, 1, [0, 0, 255], -1)

										cv.line(crop_img, start, end, [0, 255, 0], 2)

						if count_defects == 1:
										cv.putText(img, "2 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
										print("2 fingers detected")
						elif count_defects == 2:
										cv.putText(img, "3 fingers", (5, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
										print("3 fingers detected")
						elif count_defects == 3:
										cv.putText(img, "4 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
										print("4 fingers detected")
						elif count_defects == 4:
										cv.putText(img, "5 fingers", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
										print("5 fingers detected")
						elif count_defects == 0:
										cv.putText(img, "one", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
										print("ONE fingers detected")
else:
		print("No contours found")

all_img = np.hstack((drawing, crop_img))
cv.namedWindow('Test', cv.WINDOW_NORMAL)
print("Before imshow")
print(all_img)
cv.imshow('Test', all_img)
print("After imshow")
k = cv.waitKey(0)
if k == 5:
    cap.release()
    cv.destroyAllWindows()
    sys.exit()
