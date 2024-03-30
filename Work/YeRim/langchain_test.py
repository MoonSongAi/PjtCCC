import streamlit as st
from langchain_integration import setup_langchain

# Streamlit 앱의 메인 페이지 설정
st.title("LangChain Chatbot Demo")

# 사용자 입력을 받는 사이드바 설정
with st.sidebar:
    st.header("Settings")
    uploaded_files = st.file_uploader("Upload Documents", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=900, step=50)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=50, max_value=500, value=100, step=50)
    device_option = st.selectbox("Device", ['cpu', 'cuda'])
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    model_name = st.selectbox("Model Name", ['gpt-3.5-turbo', 'gpt-3', 'gpt-4', 'davinci-codex', 'curie'])

if uploaded_files and openai_api_key:
    tab = st.empty()
    # LangChain 챗봇 설정 및 초기화
    conversation_chain = setup_langchain(st, tab, uploaded_files, chunk_size, chunk_overlap, device_option, openai_api_key, model_name)
    
    if conversation_chain is not None:
        # 사용자의 질문을 입력받습니다.
        user_query = st.text_input("Your Question:")
        
        if user_query:
            with st.spinner("Fetching your answer..."):
                # 질문에 대한 답변을 생성합니다.
                response = conversation_chain({"question": user_query})
                answer = response['answer']
                st.write("Answer:", answer)
