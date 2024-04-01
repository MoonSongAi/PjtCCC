# image_processing.py에서 process_image 함수의 수정 버전
import cv2
import numpy as np
from PIL import Image

def process_image_with_hsv_range(pil_image, lower_hsv, upper_hsv):
    """
    PIL 이미지 객체와 사용자 정의 HSV 색상 범위를 받아 이미지에서 해당 색상을 강조하고,
    결과를 PIL 이미지 객체로 반환합니다.
    """
    image_np = np.array(pil_image.convert('RGB'))  # PIL 이미지를 NumPy 배열로 변환
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    result = cv2.bitwise_and(image_np, image_np, mask=mask)

    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # NumPy 배열을 PIL 이미지로 변환하여 반환
