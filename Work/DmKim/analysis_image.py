# analysis_image
# import matplotlib.pyplot as plt
import io
import os
import uuid

import fitz  # PyNuPDF
from PIL import Image
from streamlit_cropper import st_cropper


def load_to_image(uploaded_Image):
    if uploaded_Image.type == "application/pdf":
        UPLOAD_DIRECTORY = ".\Images"

        file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_Image.name)
        pdf_file = fitz.open(file_path)
        page = pdf_file.load_page(0)  # pdf page가 한페이지 인경우
        # 이미지의 해상도를 높이기 위한 matrix 설정
        matrix = fitz.Matrix(4, 4)  # 4배 확대
        pix = page.get_pixmap(matrix=matrix)  # matrix 매개변수 사용
        # pix = page.get_pixmap()
        img_bytes = pix.tobytes("ppm")  # 이미지 데이터를 PPM 형식으로 변환
        # PPM 데이터를 PIL 이미지로 변환
        img = Image.open(io.BytesIO(img_bytes))
        # print(file_path)
    else:
        img = Image.open(uploaded_Image)

    return img


def save_image_to_folder(img, folder_path="C:\\PjtCCC\\CroppedImage"):
    # 폴더가 존재하지 않는 경우 생성
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # 유니크한 파일 이름 생성 (예: 5f47b3f2-4a89-4d9c-8277-25f5b5a9f1b7.png)
    unique_filename = str(uuid.uuid4()) + ".jpg"
    # 이미지 저장 경로
    file_path = os.path.join(folder_path, unique_filename)
    # 이미지 저장
    img.save(file_path)
    print(f"Image saved to {file_path}")
    return file_path


# 예시 사용법
# img = PIL.Image.open('path_to_your_image.png')
# save_image_to_folder(img)


def analysis_image_process(st, tab, uploaded_Image):
    # Canvas 설정
    stroke_width = 5
    stroke_color = "#ff0000"  # 붉은색
    with tab:
        img = Image.open(uploaded_Image)
        cropped_img = st_cropper(
            img, realtime_update=True, box_color="#0000FF", aspect_ratio=(2, 3)
        )

        # Manipulate cropped image at will
        st.write("Preview")
        # _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)
