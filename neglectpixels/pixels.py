import cv2
import tifffile
import numpy as np
    
image = tifffile.imread('0.tif')
image2 = tifffile.imread('2.tif')
image3 = tifffile.imread('3.tif')
image4 = tifffile.imread('4.tif')
image5 = tifffile.imread('5.tif')
image6 = tifffile.imread('6.tif')
image14 = tifffile.imread('14.tif')
image29 = tifffile.imread('29.tif')

def pixels(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    record = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(record, [largest_contour], -1, 255, thickness=cv2.FILLED)
    roi_image = cv2.bitwise_and(image, image, mask=record)
    return np.count_nonzero(record == 0)

print(f"black_pixels_outside: {pixels(image)}")
print(f"black_pixels_outside: {pixels(image2)}")
print(f"black_pixels_outside: {pixels(image3)}")
print(f"black_pixels_outside: {pixels(image4)}")
print(f"black_pixels_outside: {pixels(image5)}")
print(f"black_pixels_outside: {pixels(image6)}")
print(f"black_pixels_outside: {pixels(image14)}")
print(f"black_pixels_outside: {pixels(image29)}")




