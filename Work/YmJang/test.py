import cv2
import numpy as np
from tkinter import filedialog
from tkinter import Tk

# 사용자로부터 이미지 파일 선택 받기
def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global img, hsv
    if event == cv2.EVENT_RBUTTONDOWN:
        pixel_color = hsv[y, x]
        print("Clicked HSV Color: ", pixel_color)
        
         # Hue 범위를 넓힙니다.
        hue_range = 10
        # Saturation과 Value 범위도 조정할 수 있습니다.
        saturation_range = 40
        value_range = 50
        
        lower = np.array([max(pixel_color[0] - hue_range, 0), max(pixel_color[1] - saturation_range, 0), max(pixel_color[2] - value_range, 0)])
        upper = np.array([min(pixel_color[0] + hue_range, 179), min(pixel_color[1] + saturation_range, 255), min(pixel_color[2] + value_range, 255)])

        print("Lower bound: ", lower)
        print("Upper bound: ", upper)

        mask = cv2.inRange(hsv, lower, upper)
        
        # # 검은색 선 부분만 블러 처리
        blurred_img = cv2.GaussianBlur(img, (7, 7), 0)
        blur_with_mask = img.copy()
        blur_with_mask[np.where(mask != 0)] = blurred_img[np.where(mask != 0)]

        # 그레이스케일로 변환 후 이진화
        gray = cv2.cvtColor(blur_with_mask, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 결과 표시
        cv2.imshow('Mask', mask)
        cv2.imshow('Blurred with Mask', blur_with_mask)
        cv2.imshow('Binary', binary)

        print("Selected HSV Color:", pixel_color)
        print("Lower bound:", lower)
        print("Upper bound:", upper)

image_file = select_image()
img = cv2.imread(image_file)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()