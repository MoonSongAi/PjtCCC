import json
from datetime import datetime
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# langchain 설정 및 초기화
def setup_langchain(st, tab, uploaded_files, chunk_size, chunk_overlap, device_option, openai_api_key, model_name):
    documents = get_text(uploaded_files)
    display_document_page(tab, documents)
    text_chunks = get_text_chunks(documents, chunk_size, chunk_overlap)
    save_chunks_to_json(text_chunks, 'text_chunks')
    vectorstore = get_vectorstore(text_chunks, device_option)
    conversation_chain = get_conversation_chain(vectorstore, openai_api_key, model_name)
    return conversation_chain

# 스트림릿 탭에 문서 내용 표시
def display_document_page(tab, documents):
    first_source = ''
    for doc in documents:
        # 'metadata'와 'text' 속성에 직접 접근
        metadata = getattr(doc, 'metadata', {})
        source = metadata.get('source', 'Unknown source')
        page = metadata.get('page', 'Unknown page')
        text_content = getattr(doc, 'text', 'No text available')

        if first_source != source:
            tab.subheader(f'Source: {source}')
            first_source = source
        tab.markdown(text_content.replace('\n', '<br>'), unsafe_allow_html=True)
        tab.caption(f'Page No: {page}')


# 업로드된 파일로부터 텍스트 추출
def get_text(uploaded_files):
    doc_list = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        with open(file_name, "wb") as file:
            file.write(uploaded_file.getvalue())
        logger.info(f"{file_name} 업로드 완료")
        loader = get_loader_for_file(file_name)
        if loader:
            loader_instance = loader(file_name)
            documents = loader_instance.load_and_split()
            doc_list.extend(documents)
        else:
            logger.error(f"{file_name}에 대한 적합한 로더가 없습니다.")
    return doc_list

# 파일 유형에 따른 로더 반환
def get_loader_for_file(file_name):
    if '.pdf' in file_name:
        return PyPDFLoader
    elif '.docx' in file_name:
        return Docx2txtLoader
    elif '.pptx' in file_name:
        return UnstructuredPowerPointLoader
    return None

# 텍스트 청크 생성
def get_text_chunks(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=tiktoken_len)
    chunks = []
    for doc in documents:
        text = getattr(doc, 'text', '')
        if text:
            chunks.extend(text_splitter.split_document(text))
    return chunks

# 텍스트 청크를 JSON 파일로 저장
def save_chunks_to_json(chunks, base_file_name):
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{base_file_name}_{timestamp}.json"
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)
    logger.info(f"{file_name}에 데이터 저장 완료")

# 벡터 스토어 생성
def get_vectorstore(text_chunks, device_option):
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': device_option}, encode_kwargs={'normalize_embeddings': True})
    return FAISS.from_documents(text_chunks, embeddings)

# 대화 체인 초기화 및 반환
def get_conversation_chain(vectorstore, openai_api_key, model_name='gpt-3.5-turbo'):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=model_name, temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=vectorstore.as_retriever(search_type='mmr', verbose=True), 
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), 
        get_chat_history=lambda h: h, 
        return_source_documents=True,
        verbose=True
    )
    return conversation_chain

# 텍스트의 토큰 길이를 계산하는 함수
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)
