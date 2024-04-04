# analysis_image
# import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np
import cv2

import fitz #PyNuPDF
import os
import io
import uuid
import base64

def load_to_image(uploaded_Image,pdf_value):
    if uploaded_Image.type == 'application/pdf':
        UPLOAD_DIRECTORY = ".\Images"

        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_Image.name)
        pdf_file = fitz.open(file_path)
        page = pdf_file.load_page(0)   #pdf page가 한페이지 인경우
        # 이미지의 해상도를 높이기 위한 matrix 설정
        matrix = fitz.Matrix(pdf_value, pdf_value)  # pdf_value배 확대
        pix = page.get_pixmap(matrix=matrix)  # matrix 매개변수 사용
        # pix = page.get_pixmap()
        img_bytes  = pix.tobytes("ppm") # 이미지 데이터를 PPM 형식으로 변환
        # PPM 데이터를 PIL 이미지로 변환
        img = Image.open(io.BytesIO(img_bytes))
        # print(file_path)
    else:
        img = Image.open(uploaded_Image)

    return img

def save_image_to_folder(img, folder_path='C:\\PjtCCC\\CroppedImage'):
    # 폴더가 존재하지 않는 경우 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # 유니크한 파일 이름 생성 (예: 5f47b3f2-4a89-4d9c-8277-25f5b5a9f1b7.png)
    unique_filename = str(uuid.uuid4()) + '.jpg' 
    # 이미지 저장 경로
    file_path = os.path.join(folder_path, unique_filename)
    # 이미지 저장
    img.save(file_path)
    print(f"Image saved to {file_path}")
    return file_path

# 예시 사용법
# img = PIL.Image.open('path_to_your_image.png')
# save_image_to_folder(img)

def analysis_image_process(st,tab,uploaded_Image):
    # Canvas 설정
    stroke_width = 5
    stroke_color = "#ff0000"  # 붉은색
    with tab:
        img = Image.open(uploaded_Image)
        cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                            aspect_ratio=(2,3))
                
                # Manipulate cropped image at will
        st.write("Preview")
            # _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)

def get_image_base64(image):
    # PIL Image 객체를 Base64 인코딩된 문자열로 변환
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # 이미지 포맷에 따라 "JPEG" 등으로 변경 가능
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return "data:image/png;base64," + img_str


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
