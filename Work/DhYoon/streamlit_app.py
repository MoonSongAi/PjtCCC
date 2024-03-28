# streamlit_app.py
import streamlit as st
# from streamlit_chat import message
from langchain_community.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

from langchain_integration import setup_langchain
from analysis_image import analysis_image_process

def main():
    st.set_page_config(
        page_title="무었이든",
        page_icon=":volcano:")

    st.title("_무었이 불편 하실까? :red[QA Chat]_ :volcano:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    tab1 , tab2 ,tab3 = st.tabs(["🧑‍🚀chat.....","🕵️‍♂️ chucked Data","💫Image processing...."])

    with st.sidebar:
        with st.expander("Select Image",expanded=True):            
            uploaded_Image =  st.file_uploader("Select target Image",type=['png','jpg'],accept_multiple_files=True)
            # 파일이 업로드 되었는지 확인하고 버튼의 활성화 상태 결정
            button_enabled = uploaded_Image is not None and len(uploaded_Image) > 0
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
    if process_image:
        analysis_image_process(st,tab3,uploaded_Image)

    if process_lang:
        if not openai_api_key:
            openai_api_key = st.secrets["OpenAI_Key"]
            if not openai_api_key:
                st.info("Please add your OpenAI API key to continue.")
                st.stop()
        # Langchain 설정
        conversation_chain = setup_langchain(st , tab2, 
                                            uploaded_files,
                                            chunk_size,chunk_overlap,device_option,
                                            openai_api_key,model_name)

        st.session_state.conversation = conversation_chain

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]

    for message in st.session_state.messages:
        with tab1.chat_message(message["role"]):
            tab1.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    query = tab1.chat_input("질문을 입력해주세요.")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with tab1.chat_message("user"):
            tab1.markdown(query)

        with tab1.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents = result['source_documents']

                tab1.markdown(response)
                with tab1.expander("참고 문서 확인"):
                    tab1.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    tab1.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    tab1.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)


if __name__ == '__main__':
    main()
