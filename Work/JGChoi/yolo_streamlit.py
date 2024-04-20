import streamlit as st
from PIL import Image
import torch
import numpy as np
import os

# 모델 로드 함수
@st.cache_resource
def load_custom_model(model_path):
    if os.path.exists(model_path):
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        st.write("모델 로드 성공!")
        return model
    else:
        st.write("모델 파일 경로가 유효하지 않습니다.")
        return None

# 객체 인식 함수
def run_object_detection(model, image):
    results = model(image)
    return results

# Streamlit 애플리케이션 구성
def main():
    st.title("YOLOv5 사용자 정의 모델로 객체 인식")
    
    model_path = st.text_input("모델 파일 경로 입력", 'C:\\PjtCCC\\yolov5\\runs\\train\\runs\\result\\weights\\best.pt')
    if st.button("모델 로드"):
        model = load_custom_model(model_path)
    else:
        model = None

    uploaded_file = st.file_uploader("이미지 업로드", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None and model is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='업로드한 이미지', use_column_width=True)
        
        if st.button('객체 인식 실행'):
            img_array = np.array(image)
            results = run_object_detection(model, [img_array])
            if results:
                results.render()  # 결과 렌더링
                for img in results.imgs:
                    result_img = Image.fromarray(img)
                    st.image(result_img, caption='객체 인식 결과', use_column_width=True)
            else:
                st.write("객체를 인식하지 못했습니다.")

if __name__ == '__main__':
    main()



## C:\\PjtCCC\\yolov5\\runs\\train\\runs\\result\\weights\\best.pt
## streamlit run C:\PjtCCC\Work\JGChoi\yolo_streamlit.py
