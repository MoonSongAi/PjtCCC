import cv2
import numpy as np
from streamlit_cropper import st_cropper

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 마우스 버튼 클릭 이벤트
        # 선택한 위치의 HSV 색상 추출
        pixel_color = hsv[y, x]
        lower = np.array([pixel_color[0] - 10, 100, 100])
        upper = np.array([pixel_color[0] + 10, 255, 255])

        # 추출된 색상에 해당하는 마스크 생성
        mask = cv2.inRange(hsv, lower, upper)

        # 클로징 연산으로 작은 간격을 메꿀 수 있도록 마스크 처리
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 마스크를 사용하여 이미지 상의 해당 색상을 제거
        img_result = cv2.bitwise_and(img, img, mask=~mask_closed)

        # 검은색 선 부분만 블러 처리를 합니다.
        # blurred_img = cv2.GaussianBlur(img, (7, 7), 0)
        # img[np.where(mask != 0)] = blurred_img[np.where(mask != 0)]

        # 결과 이미지를 그레이스케일로 변환하고 이진화
        gray = cv2.cvtColor(img_result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 이진화된 이미지와 원본 이미지를 표시
        cv2.imshow('Binary Image', thresh)
        # cv2.imshow('Original Image', img)
        # cv2.imshow('Masked Image', img_result)

# Streamlit 앱에서 이미지를 저장하고 그 경로를 반환받는 부분
saved_image_path = save_image_to_folder(cropped_img)

# 이제 OpenCV 코드에서 saved_image_path를 사용하여 이미지를 불러옵니다.
img = cv2.imread(saved_image_path)
if img is None:
    print("이미지를 불러오는데 실패했습니다. 경로를 확인하세요.")
else:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 윈도우 생성 및 마우스 콜백 함수 설정
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    # 이미지 표시
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()