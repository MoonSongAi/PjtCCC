# streamlit_app.py
import streamlit as st
# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from image_processing import process_image_with_hsv_range  # 함수 이름 변경 및 인자 추가

from langchain_integration import setup_langchain
from analysis_image import save_image_to_folder ,load_to_image

from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np

import os


def main():
    st.set_page_config(
        page_title="표시 디자인",
        page_icon=":volcano:")


    st.title("_표시 디자인 오류 탐색기..._ :red[QA Chat]_ :volcano:")
    # 여기에 CSS 스타일을 추가
    st.markdown("""
        <style>
        /* 여기에 CSS 스타일을 추가 */
        #tabs-bui3-tab-0>.st-emotion-cache-l9bjmx p,
        #tabs-bui3-tab-1>.st-emotion-cache-l9bjmx p,
        #tabs-bui3-tab-2>.st-emotion-cache-l9bjmx p{
            /* 탭 아이템 스타일 변경 */
            font-size:25px
        }
        .element-container iframe{
                border:3px dashed black
        }
            
        .st-emotion-cache-1kyxreq div{
                border:3px dashed red
        } 
        </style>
    """, unsafe_allow_html=True)
    

    tab1 , tab2 ,tab3 = st.tabs(["💫Image processing....","🧑‍🚀chat about Design....","🕵️‍♂️ chucked Data"])

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    # 1단계: 초기 상태 설정
    if 'canvas_image_data' not in st.session_state:
        st.session_state.canvas_image_data = None
    # 회전된 이미지를 저장할 session_state 초기화
    if 'rotation_angle' not in st.session_state:
        st.session_state.rotation_angle = 0
    if 'saved_images' not in st.session_state:
        st.session_state.saved_images = []
    if 'images_list' not in st.session_state:
        st.session_state.images_list = []
    if 'loaded_image' not in st.session_state:
        st.session_state.loaded_image = None
    
    # color_options 딕셔너리 정의
    color_options = {
        "Red": ([0, 100, 100], [10, 255, 255]),
        "Green": ([50, 100, 100], [70, 255, 255]),
        "Blue": ([110, 100, 100], [130, 255, 255]),
        "Yellow": ([25, 100, 100], [35, 255, 255]),
        "Black": ([0, 0, 0], [180, 255, 30])  # HSV 범위 예시 수정
    }

    with st.sidebar:
        with st.expander("Select Image",expanded=True):            
            uploaded_Image =  st.file_uploader("Select target Image",type=['pdf','png','jpg'])
            # 파일이 업로드 되었는지 확인하고 버튼의 활성화 상태 결정
                # pDF image 해상
            pdf_value = st.slider(
                    label='PDF Resolution',  # 슬라이더 라벨
                    min_value=1,  # 최소값
                    max_value=10,  # 최대값
                    value=2,  # 기본값
                    step=1  # 단계
            )
            button_enabled = uploaded_Image is not None
            # 색상 선택 드롭다운 메뉴
            selected_color_name = st.selectbox("이미지 전처리 color_options", options=list(color_options.keys()))
            lower_hsv, upper_hsv = color_options[selected_color_name]
            process_image = st.button("Analysis Design file....", disabled=not button_enabled)


        with st.expander("Setting for LangChain",expanded=False):            
            uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
            # Streamlit 사이드바에 슬라이더 추가
            openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=900, step=50)
            chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=100, step=10)
            # Streamlit 사이드바 콤보박스 추가
            device_option = st.selectbox(
                "Choose the device for the model",
                options=['cpu', 'cuda', 'cuda:0', 'cuda:1'],  # 여기에 필요한 모든 옵션을 추가하세요.
                index=0  # 'cpu'를 기본값으로 설정
            )
            # Streamlit 사이드바  콤보박스 추가
            model_name = st.selectbox(
                "Choose the model for OpenAI LLM API",
                options=['gpt-3.5-turbo', 'gpt-3', 'gpt-4','davinci-codex', 'curie'],  # 사용 가능한 모델 이름들
                index=0  # 'gpt-3.5-turbo'를 기본값으로 설정
            )
            # 파일이 업로드 되었는지 확인하고 버튼의 활성화 상태 결정
            button_enabled = uploaded_files is not None and len(uploaded_files) > 0
            process_lang = st.button("Process....", disabled=not button_enabled)
    if uploaded_Image:
        if st.session_state.loaded_image != uploaded_Image:    #image가 변경되었다면
            st.session_state.loaded_image = uploaded_Image
            st.session_state.saved_images = []
            st.session_state.images_list = []
        with tab1:
            img = load_to_image(uploaded_Image,pdf_value)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000',
                                                    aspect_ratio=(1,1))
                        
            # Manipulate cropped image at will
            st.session_state.canvas_image_data = cropped_img
            # _ = cropped_img.thumbnail((300,300))

            # 버튼 배치를 위한 컬럼 생성
            col1, col2  = st.columns(2)
            with col1:
                # 이미지 저장 버튼
                save_image = st.button("Save cropped image")
            with col2:
                # 이미지 회전 버튼
                rotate_image = st.button("Rotate cropped image")

            if rotate_image:
                st.session_state.rotation_angle += 90   # 회전 각도 업데이트
                st.session_state.rotation_angle %= 360  # 360도가 되면 0으로 리셋
                # 현재 회전 각도에 따라 이미지 회전
            
            st.session_state.canvas_image_data = st.session_state.canvas_image_data.rotate(
                                                              st.session_state.rotation_angle,
                                                              expand = True)

            st.write("***_:blue[Preview Cropped Image]_***")
            st.image(st.session_state.canvas_image_data)
            if save_image:
                save_name = save_image_to_folder(st.session_state.canvas_image_data)
                # 저장된 이미지 리스트에 이미지 추가
                st.session_state.saved_images.append(st.session_state.canvas_image_data)
                st.session_state.images_list.append(save_name)
                st.session_state.rotation_angle = 0
            # 저장된 이미지 썸네일을 횡으로 나열하여 표시
            if st.session_state.saved_images:
                # 각 이미지를 작은 썸네일로 변환하여 표시
                cols = st.columns(len(st.session_state.saved_images))
                for idx, saved_image in enumerate(st.session_state.saved_images):
                    with cols[idx]:
                        # 썸네일 크기로 이미지 리사이즈
                        st.caption(st.session_state.images_list[idx])
                        saved_image.thumbnail((200, 200))
                        st.image(saved_image, width=100)  # 썸네일 이미지 표시

######################## 추가함 ###################
    
    selected_color_name = st.sidebar.selectbox("색상 선택", list(color_options.keys()))
    lower_hsv, upper_hsv = color_options[selected_color_name]
    
    # "모든 이미지 처리" 버튼을 메인 페이지에 배치
    if st.button('이미지 전처리'):
        # 처리된 이미지를 저장할 리스트 초기화
        if 'processed_images' not in st.session_state:
            st.session_state.processed_images = []
        
        # 이미지 처리 및 저장
        if 'images_list' in st.session_state:
            for image_name in st.session_state.images_list:
                image_path = os.path.join('path_to_saved_images', image_name)
                pil_image = Image.open(image_path).convert('RGB')
                processed_image = process_image_with_hsv_range(pil_image, np.array(lower_hsv), np.array(upper_hsv))
                
                # 처리된 이미지를 session_state에 추가
                st.session_state.processed_images.append(processed_image)
        
        # 처리된 이미지를 썸네일 형태로 표시
        if 'processed_images' in st.session_state and st.session_state.processed_images:
            cols = st.columns(len(st.session_state.processed_images))
            for idx, processed_image in enumerate(st.session_state.processed_images):
                with cols[idx]:
                    # 썸네일 이미지 표시 (PIL.Image 객체를 사용하는 경우)
                    st.caption(st.session_state.images_list[idx])
                    processed_image.thumbnail((200, 200))
                    st.image(processed_image, width=100)
        else:
            st.write("이미지가 없습니다.")

###################################################

    if process_lang:
        if not openai_api_key:
            openai_api_key = st.secrets["OpenAI_Key"]
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
        # Langchain 설정
        conversation_chain = setup_langchain(st , tab3, 
                                            uploaded_files,
                                            chunk_size,chunk_overlap,device_option,
                                            openai_api_key,model_name)

        st.session_state.conversation = conversation_chain

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with tab2.chat_message(message["role"]):
            tab2.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    query = tab2.chat_input("질문을 입력해주세요.")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with tab2.chat_message("user"):
            tab2.markdown(query)

        with tab2.chat_message("assistant"):
            chain = st.session_state.conversation
            if chain is None:
                st.warning('학습된 정보가 없습니다.')
                st.stop()

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                tab2.markdown(response)
                with tab2.expander("참고 문서 확인"):
                    tab2.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    tab2.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    tab2.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)


if __name__ == '__main__':
    main()