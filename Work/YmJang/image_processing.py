import cv2
import numpy as np
from PIL import Image

def process_image_with_hsv_range(pil_image, lower_hsv, upper_hsv):
    """
    PIL 이미지 객체와 사용자 정의 HSV 색상 범위를 받아 이미지에서 해당 색상을 강조하고,
    선택된 색상 부분만 블러 처리한 후, 그레이스케일로 변환하여 이진화를 수행합니다.
    결과를 PIL 이미지 객체로 반환합니다.
    """
    image_np = np.array(pil_image.convert('RGB'))  # PIL 이미지를 NumPy 배열로 변환
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # 선택된 색상 부분만 블러 처리
    blurred_img = cv2.GaussianBlur(image_np, (7, 7), 0)
    image_np[np.where(mask != 0)] = blurred_img[np.where(mask != 0)]

    # 그레이스케일로 변환 후 이진화
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 이진화 결과를 PIL 이미지로 변환하여 반환
    return Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB))
