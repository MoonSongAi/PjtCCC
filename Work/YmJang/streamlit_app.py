# streamlit_app.py
import streamlit as st
# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory
from langchain_integration import is_vector_db ,load_langchain,  setup_langchain
from analysis_image import save_image_to_folder ,load_to_image , \
                        get_image_base64,process_image_with_hsv_range 
from text_detection_comparison import TextDetectionAndComparison
from streamlit_cropper import st_cropper
from OCR_anal import specialDicForOCR , execute_OCR
from PIL import Image
import numpy as np
import os


# 이미지 삭제 함수
def delete_image(image_index):
    if 0 <= image_index < len(st.session_state.saved_images):
        del st.session_state.saved_images[image_index]
        del st.session_state.images_list[image_index]
        del st.session_state.process_images[image_index]
        del st.session_state.process_list[image_index]
        # 이미지 리스트 변경 후 다시 이미지와 버튼 표시를 위해 페이지 갱신
        if len(st.session_state.images_list) == 0:
            st.session_state.anal_button_click = False
        st.rerun()

def main():
    # Google Cloud 자격 증명 파일의 경로를 사용하여 클래스 초기화
    # detector = TextDetectionAndComparison("C:\\keys\\feisty-audio-420101-460dfe33e2cb.json")
    
    # OCR을 위한 사전 준비 작업
    myDic = specialDicForOCR()

    DB_INDEX = "VECTOR_DB_INDEX"
    st.set_page_config(
        page_title="표시 디자인",
        page_icon=":volcano:")


    st.title("_표시 디자인 오류 탐색기_ :red[QA Chat]:volcano:")
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
                
        .st-emotion-cache-7ym5gk{
                width:14rem;
                height:5rem
        }
        </style>
    """, unsafe_allow_html=True)
    

    tab1 , tab2 ,tab3 = st.tabs(["💫Image processing","🧑‍🚀chat about Design","🕵️‍♂️ chuncked Data"])

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
    if 'ocr_images' not in st.session_state:
        st.session_state.ocr_images = []
    if 'images_list' not in st.session_state:
        st.session_state.images_list = []
    if 'process_list' not in st.session_state:
        st.session_state.process_list = []
    if 'ocr_list' not in st.session_state:
        st.session_state.ocr_list = []
    if 'loaded_image' not in st.session_state:
        st.session_state.loaded_image = None
    if 'anal_process' not in st.session_state:
        st.session_state.anal_process = False
    if 'anal_button_click' not in st.session_state:
        st.session_state.anal_button_click = False
    if 'delete_request' not in st.session_state:
        st.session_state.delete_request = False
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = is_vector_db(DB_INDEX)   
    if 'anal_image_data' not in st.session_state:
        st.session_state.anal_image_data = [] 

    with st.sidebar:
        with st.expander("Adjust HSV Threshold",expanded=False):
            colors = ["Red", "Green", "Blue", "Yellow", "Black"]
            default_color = "Black"             # 기본으로 선택하고 싶은 색상
            default_index = colors.index(default_color)  # 'Black'의 인덱스 찾기
            color_selection = st.selectbox("Select line color", colors, index=default_index)
            if color_selection == "Red":
                # 클릭한 HSV 색상:  [174 250 209]
                lower =  [164,210,159]
                upper =  [179,255,255]
                # lower = [0, 100, 100] 
                # upper = [10, 255, 255]
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
                #클릭한 HSV 색상:  [ 89 255  84]
                lower = [79, 215, 34]
                upper = [99, 255, 134]
                # lower = [0, 0, 0]
                # upper = [180, 255, 50]
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
            print('st.expander:',st.session_state.anal_process)

            process_image = st.button("Analysis Design file....", disabled= not st.session_state.anal_process)
            if process_image:
               st.session_state.anal_button_click = True  #Button Click 을 session 동안 유지 하기위해서 

        with st.expander("Setting for LangChain",expanded=False):            
            openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
            model_name = st.selectbox(
                "Choose the model for OpenAI LLM API",
                options=['gpt-3.5-turbo', 'gpt-3', 'gpt-4','davinci-codex', 'curie'],  # 사용 가능한 모델 이름들
                index=0  # 'gpt-3.5-turbo'를 기본값으로 설정
            )

            load_lang = st.button("load vector DB", disabled= not st.session_state.vector_db)

            uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
            # Streamlit 사이드바에 슬라이더 추가
            chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=900, step=50)
            chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=100, step=10)
            # Streamlit 사이드바 콤보박스 추가
            device_option = st.selectbox(
                "Choose the device for the model",
                options=['cpu', 'cuda', 'cuda:0', 'cuda:1'],  # 여기에 필요한 모든 옵션을 추가하세요.
                index=0  # 'cpu'를 기본값으로 설정
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
            st.session_state.process_list = []
            st.session_state.ocr_images = []
            st.session_state.ocr_list = []
            st.session_state.anal_image_data = [] 
            del_buttons = []

            st.session_state.anal_process = False

        with tab1:
            img = load_to_image(uploaded_Image,pdf_value)
            cropped_img = st_cropper(img, realtime_update=True, box_color='#FF0000',
                                                    aspect_ratio=(1,1))
                        
            # Manipulate cropped image at will
            st.session_state.canvas_image_data = cropped_img
            # _ = cropped_img.thumbnail((300,300))

            col1, col2  = st.columns(2)
            with col1:
                # 이미지 회전 버튼
                rotate_image = st.button("Rotate cropped image")
            with col2:
                # 이미지 저장 버튼
                save_image = st.button("Save cropped image")
            if rotate_image:
                st.session_state.rotation_angle -= 90   # 회전 각도 업데이트
                st.session_state.rotation_angle %= 360  # 360도가 되면 0으로 리셋
                # 현재 회전 각도에 따라 이미지 회전
            st.session_state.canvas_image_data = st.session_state.canvas_image_data.rotate(
                                                              st.session_state.rotation_angle,
                                                              expand = True)
            print(f"st.session_state.rotation_angle={st.session_state.rotation_angle}")

            if save_image:
                save_name = save_image_to_folder(st.session_state.canvas_image_data)
                # 저장된 이미지 리스트에 이미지 추가
                st.session_state.saved_images.append(st.session_state.canvas_image_data)
                st.session_state.images_list.append(save_name)


            st.write("***_:blue[Preview Cropped Image]_***")
            st.image(st.session_state.canvas_image_data)
            st.session_state.anal_process = True
            # 저장된 이미지 썸네일을 횡으로 나열하여 표시
            if st.session_state.saved_images:
                # 각 이미지를 작은 썸네일로 변환하여 표시
                cols = st.columns(len(st.session_state.saved_images))
                for idx, saved_image in enumerate(st.session_state.saved_images):
                    with cols[idx]:
                        # 썸네일 크기로 이미지 리사이즈
                        thumbnail_images = st.session_state.saved_images[idx].copy()
                        st.caption(st.session_state.images_list[idx])
                        thumbnail_images.thumbnail((200, 200))
                        st.image(thumbnail_images, width=100)  # 썸네일 이미지 표시
                        # 삭제 버튼 생성
                        if st.button(f'Delete {idx}', key=f"delete_{idx}"):
                            st.session_state.delete_request = True
                            delete_image(idx)   # 삭제 버튼이 클릭되면, 해당 이미지를 삭제하기 위한 플래그 설정

            if st.session_state.anal_button_click:
                # 새로운 이미지 리스트를 처리하기 전에 기존의 Zoom In 상태를 초기화하거나 업데이트합니다.
                for idx, _ in enumerate(st.session_state.images_list):
                    zoom_key = f"zoom_{idx}"
                    # session_state에 확대 상태를 저장할 변수가 없으면 초기화합니다.
                    if zoom_key not in st.session_state:
                        st.session_state[zoom_key] = False

                # 선택된 이미지 이름으로 실제 이미지 객체를 얻음
                cols = st.columns(len(st.session_state.images_list))
                for idx, img_path in enumerate(st.session_state.images_list):
                    # print(len(st.session_state.process_list)-1,idx)
                    if len(st.session_state.process_list)-1 < idx:  #새로운 이미지가 추가된 경우만
                        image = Image.open(img_path).convert('RGB')
                        processed_image = process_image_with_hsv_range(image, lower_hsv, upper_hsv)
                    
                        # 저장된 이미지 리스트에 이미지 추가
                        st.session_state.process_images.append(processed_image)
                        save_name = save_image_to_folder(processed_image)
                        st.session_state.process_list.append(save_name)

                if len(st.session_state.process_list) > 0:
                    # print(st.session_state.process_list)
                    for idx, img_path in enumerate(st.session_state.process_list):
                        zoom_key = f"zoom_{idx}"                       
                        with cols[idx]:
                            # 화면에 표시하기 위해 썸네일 이미지를 준비합니다.
                            display_image = st.session_state.process_images[idx].copy()
                            display_image.thumbnail((200, 200))
                            st.image(display_image, width=100)  # 썸네일 이미지로 표시

                            # 'Zoom In' 버튼 클릭 시 처리
                            if st.button(f'Zoom In {idx}', key=f"zoomin_{idx}"):
                                # 확대 상태를 토글합니다.
                                st.session_state[zoom_key] = not st.session_state[zoom_key]
                            
                            # 확대 상태에 따라 이미지를 표시하거나 숨깁니다.
                            if st.session_state[zoom_key]:
                                st.image(st.session_state.process_images[idx], width=700)

                        if  len(st.session_state.anal_image_data) <= idx: 
                            st.session_state.anal_image_data.append(None)  # 이미지 처리 결과 대신 None 추가
                        # 이미지 처리 결과가 이미 있으면 사용, 없으면 새로 처리
                        if st.session_state.anal_image_data[idx] is None:
                            # 이미지 처리 함수를 호출하여 결과를 저장
                            ret_image = execute_OCR(myDic, img_path)
                            st.session_state.anal_image_data[idx] = ret_image

                        # 결과 이미지를 표시
                        if st.session_state.anal_image_data[idx] is not None:
                            st.image(st.session_state.anal_image_data[idx])

    if not openai_api_key:
       openai_api_key = st.secrets["OpenAI_Key"]
       if not openai_api_key:
          st.info("Please add your OpenAI API key to continue.")
          st.stop()

    if load_lang:
        conversation_chain = load_langchain(DB_INDEX,device_option,openai_api_key,model_name)
        st.session_state.conversation = conversation_chain
        st.session_state.processComplete = True

    if process_lang:
        conversation_chain = setup_langchain(st , tab3, 
                                            uploaded_files,
                                            chunk_size,chunk_overlap,device_option,
                                            openai_api_key,model_name)
        st.session_state.conversation = conversation_chain
        st.session_state.processComplete = True

    with tab2:
        # 버튼에 표시될 내용을 리스트로 정의
        button_labels = ["글자크기와 장평 가이드라인",
                         "원산지 표시법",
                         "굵게 표시해야하는 항목은 무엇이 있나요?",
                         "영양정보 표시할때 주의사항은?",
                         "원재료 표시 기준",
                         "정보표시면 표시방법"]
            # 2행 3열 구조로 버튼을 배치하기 위한 인덱스
        if 'last_clicked' not in st.session_state:
            st.session_state['last_clicked'] = ''

        idx = 0
        # 두 행을 생성
        for i in range(2):  # 두 행
            cols = st.columns(3)
            for col in cols:  # 각 행에 3개의 열
                if idx < len(button_labels):
                    button_key = f"button_{idx}"
                    if col.button(button_labels[idx], key=button_key):
                        # 버튼 클릭 시, 해당 버튼의 레이블을 저장
                        st.session_state['last_clicked'] = button_labels[idx]
                    idx += 1

        if 'messages' not in st.session_state:
            st.session_state['messages'] = [{"role": "assistant", 
                                            "content": "안녕하세요! 표시디자인과 관련된 궁금하신 것이 있으면 무엇이든 질문 하세요!"}]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        history = StreamlitChatMessageHistory(key="chat_messages")

        # Chat logic
        if st.session_state['last_clicked'] != '':
            query_text =  st.session_state['last_clicked']
            st.session_state['last_clicked'] = ''
            query = query_text
            st.chat_input(query_text)
        else:
            query_text = "질문을 입력해주세요."
            query = st.chat_input(query_text)

        if query:
            st.session_state.messages.append({"role": "user", "content": query})

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                chain = st.session_state.conversation
                if chain is None:
                    st.warning('학습된 정보가 없습니다.')
                    st.stop()

                with st.spinner("Thinking..."):
                    result = chain({"question": query})
                    with get_openai_callback() as cb:
                        st.session_state.chat_history = result['chat_history']
                    response = result['answer']
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    source_documents = result['source_documents']

                    st.markdown(response)
                    with st.expander("참고 문서 확인"):
                        st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                        st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                        st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)


if __name__ == '__main__':
    main()