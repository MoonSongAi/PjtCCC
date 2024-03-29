# streamlit_app.py
import streamlit as st
# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from langchain_integration import setup_langchain
from analysis_image import save_image_to_folder

from streamlit_cropper import st_cropper
from PIL import Image

import fitz #PyNuPDF
import os
import io

def main():
    st.set_page_config(
        page_title="표시 디자인",
        page_icon=":volcano:")


    st.title("_표시 디자인 오류....?_ :red[QA Chat]_ :volcano:")

    tab1 , tab2 ,tab3 = st.tabs(["💫Image processing....","🧑‍🚀chat.....","🕵️‍♂️ chucked Data"])

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None
    # 1단계: 초기 상태 설정
    if 'canvas_image_data' not in st.session_state:
        st.session_state.canvas_image_data = None
    

    with st.sidebar:
        with st.expander("Select Image",expanded=True):            
            uploaded_Image =  st.file_uploader("Select target Image",type=['pdf','png','jpg'])
            # 파일이 업로드 되었는지 확인하고 버튼의 활성화 상태 결정
            button_enabled = uploaded_Image is not None 
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
        # analysis_image_process(st,tab3,uploaded_Image)
        with tab1:
            if uploaded_Image.type == 'application/pdf':
                UPLOAD_DIRECTORY = ".\Images"

                file_path = os.path.join(UPLOAD_DIRECTORY, uploaded_Image.name)
                pdf_file = fitz.open(file_path)
                page = pdf_file.load_page(0)
                pix = page.get_pixmap()
                img_bytes  = pix.tobytes("ppm") # 이미지 데이터를 PPM 형식으로 변환
                # PPM 데이터를 PIL 이미지로 변환
                img = Image.open(io.BytesIO(img_bytes))
                # print(file_path)
            else:
                img = Image.open(uploaded_Image)

            cropped_img = st_cropper(img, realtime_update=True, box_color='#0000FF',
                                                    aspect_ratio=(1,1))
                        
            # Manipulate cropped image at will
            st.session_state.canvas_image_data = cropped_img
            # _ = cropped_img.thumbnail((300,300))
            save_image = st.button("Save cropped image")

            st.write("Cropped Image Preview")
            st.image(cropped_img)
            if save_image:
                save_image_to_folder(cropped_img)

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
