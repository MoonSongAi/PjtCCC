# analysis_image
# import matplotlib.pyplot as plt
from streamlit_cropper import st_cropper
from PIL import Image

import os
import uuid

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

