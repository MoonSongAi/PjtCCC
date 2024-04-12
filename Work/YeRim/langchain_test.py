import streamlit as st
from langchain_integration import setup_langchain, load_langchain, is_vector_db

def main():
    st.title("LangChain Integration Test")

    # 초기 상태 설정
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = is_vector_db("VECTOR_DB_INDEX")
    
    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        model_name = st.selectbox(
            "Choose the model for OpenAI LLM API",
            options=['gpt-3.5-turbo', 'gpt-3', 'gpt-4', 'davinci-codex', 'curie'],
            index=0
        )
        device_option = st.selectbox(
            "Choose the device for the model",
            options=['cpu', 'cuda', 'cuda:0', 'cuda:1'],
            index=0
        )

        uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx'], accept_multiple_files=True)

        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=900, step=50)
        chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=100, step=10)

        load_langchain_button = st.button("Load LangChain", disabled=not st.session_state.vector_db)
        setup_langchain_button = st.button("Setup LangChain", disabled=not uploaded_files)
    
    # Load LangChain with existing vector database
    if load_langchain_button:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key.")
        else:
            conversation_chain = load_langchain("VECTOR_DB_INDEX", device_option, openai_api_key, model_name)
            st.session_state.conversation = conversation_chain
            st.success("LangChain loaded successfully.")

    # Setup LangChain with new documents
    if setup_langchain_button:
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key.")
        else:
            conversation_chain = setup_langchain(st, st, uploaded_files, chunk_size, chunk_overlap, device_option, openai_api_key, model_name)
            st.session_state.conversation = conversation_chain
            st.success("LangChain setup complete with uploaded documents.")

    # Chat interface
    if 'conversation' in st.session_state and st.session_state.conversation is not None:
        st.header("Ask a question")
        user_input = st.text_input("Enter your question here:")
        
        if user_input:
            with st.spinner("Finding an answer..."):
                conversation_chain = st.session_state.conversation
                result = conversation_chain({"question": user_input})
                response = result.get('answer', "Sorry, I can't find an answer for that.")
                st.write(response)

if __name__ == "__main__":
    main()
