# streamlit_app.py
import streamlit as st
# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from langchain_integration import setup_langchain
from analysis_image import save_image_to_folder ,load_to_image ,process_image_with_hsv_range

from streamlit_cropper import st_cropper
from PIL import Image
import numpy as np



import os

def delete_image(index):
    # 주어진 인덱스의 이미지와 캡션을 삭제
    del st.session_state.saved_images[index]
    del st.session_state.images_list[index]

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
    if 'process_images' not in st.session_state:
        st.session_state.process_images = []
    if 'images_list' not in st.session_state:
        st.session_state.images_list = []
    if 'loaded_image' not in st.session_state:
        st.session_state.loaded_image = None
    if 'anal_image' not in st.session_state:
        st.session_state.anal_image = False
    

    with st.sidebar:
        with st.expander("Adjust HSV Threshold",expanded=False):
            color_selection = st.selectbox("Select Color", ["Red", "Green", "Blue","Yellow","Black"])
            if color_selection == "Red":
                lower = [0, 100, 100] 
                upper = [10, 255, 255]
            elif color_selection == "Green":
                lower = [40,40,40]
                upper = [80,255,255]
            elif color_selection == "Blue":
                lower = [40,40,40]
                upper = [80,255,255]
            elif color_selection == "Green":
                lower = [40,40,40]
                upper = [80,255,255]
            elif color_selection == "Black":
                lower = [0, 0, 0]
                upper = [180, 255, 50]
            else:
                lower = [0,100,100]
                upper = [10,255,255]

            lower_h = st.slider('Lower Hue', 0, 179, lower[0])
            lower_s = st.slider('Lower Saturation', 0, 255, lower[1])
            lower_v = st.slider('Lower Value', 0, 255, lower[2])
            upper_h = st.slider('Upper Hue', 0, 179, upper[0])
            upper_s = st.slider('Upper Saturation', 0, 255, upper[1])
            upper_v = st.slider('Upper Value', 0, 255, upper[2])

            lower_hsv = np.array([lower_h, lower_s, lower_v])
            upper_hsv = np.array([upper_h, upper_s, upper_v])

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
            print('st.expander:',st.session_state.anal_image)

            process_image = st.button("Analysis Design file....", disabled= not st.session_state.anal_image)


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
            st.sidebar.write("New image loading...........")
            st.session_state.loaded_image = uploaded_Image
            st.session_state.saved_images = []
            st.session_state.images_list = []
            st.session_state.process_images = []
            del_buttons = []

            st.session_state.anal_image = False

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
            # print(f"st.session_state.rotation_angle={st.session_state.rotation_angle}")

            st.write("***_:blue[Preview Cropped Image]_***")
            st.image(st.session_state.canvas_image_data)
            st.session_state.anal_image = True
            if save_image:
                save_name = save_image_to_folder(st.session_state.canvas_image_data)
                # 저장된 이미지 리스트에 이미지 추가
                st.session_state.saved_images.append(st.session_state.canvas_image_data)
                st.session_state.images_list.append(save_name)
                st.session_state.rotation_angle = 0
                st.sidebar.write("Save image loading...........")
                # print('save_image:',st.session_state.anal_image)

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
                        # 삭제 버튼 생성
                        if st.button('Delete'+str(idx), key="delete"+str(idx)):
                            del_buttons.append(idx)
                        # 삭제 버튼이 클릭된 경우
                for idx in reversed(del_buttons):  # 뒤에서부터 삭제해야 인덱스 문제가 발생하지 않음
                    delete_image(idx)
    if process_image:
        # 선택된 이미지 이름으로 실제 이미지 객체를 얻음
        cols = st.columns(len(st.session_state.images_list))
        for idx, img_path in enumerate(st.session_state.images_list):
            image = Image.open(img_path).convert('RGB')
            processed_image = process_image_with_hsv_range(image, lower_hsv, upper_hsv)
            st.session_state.process_images.append(processed_image)
            print(lower_hsv,upper_hsv)
            with cols[idx]:
                # 썸네일 크기로 이미지 리사이즈
                st.caption(img_path)
                processed_image.thumbnail((200, 200))
                st.image(processed_image, width=100)  # 썸네일 이미지 표시
                st.button("확대"+str(idx))

    else:
        print("process_image False")

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
