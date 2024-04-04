import json
from datetime import datetime
import numpy as np
import tiktoken
from loguru import logger

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_langchain(st, tab, uploaded_files, chunk_size, chunk_overlap, device_option, openai_api_key, model_name):
    logger.add("logger.log", rotation="10 MB")
    documents = get_text(uploaded_files)
    display_document_page(tab, documents)
    text_chunks = get_text_chunks(documents, chunk_size, chunk_overlap)
    embeddings_model = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': device_option}, encode_kwargs={'normalize_embeddings': True})
    save_vectors_to_json(text_chunks, embeddings_model, 'text_vectors')

def display_document_page(tab, documents):
    first_source = ''
    for doc in documents:
        metadata = getattr(doc, 'metadata', {})
        source = metadata.get('source', 'Unknown source')
        page = metadata.get('page', 'Unknown page')

        # 안전하게 'text' 속성에 접근
        text_content = getattr(doc, 'text', 'No text available')

        if first_source != source:
            tab.subheader(f'Source: {source}')
            first_source = source
        tab.markdown(text_content.replace('\n', '<br>'), unsafe_allow_html=True)
        tab.caption(f'Page No: {page}')

        
def get_text(uploaded_files):
    doc_list = []
    for uploaded_file in uploaded_files:
        file_path = f"./{uploaded_file.name}"
        try:
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getvalue())
            logger.info(f"Uploaded {uploaded_file.name}")
            loader_instance = select_loader(file_path)
            if loader_instance:
                documents = loader_instance.load_and_split()
                doc_list.extend(documents)
            else:
                logger.error(f"No suitable loader found for {uploaded_file.name}.")
        except Exception as e:
            logger.error(f"Failed to process {uploaded_file.name}: {e}")
    return doc_list

def select_loader(file_path):
    if file_path.endswith('.pdf'):
        return PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        return Docx2txtLoader(file_path)
    elif file_path.endswith('.pptx'):
        return UnstructuredPowerPointLoader(file_path)

def get_text_chunks(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=tiktoken_len)
    text_chunks = []
    for doc in documents:
        text = getattr(doc, 'text', '')
        if text:
            chunks = text_splitter.split_document(text)
            text_chunks.extend(chunks)
            for chunk in chunks:
                logger.info(f"Chunk length: {len(chunk)}, Content: {chunk[:50]}...")  # 청크의 내용 일부 로깅
    if not text_chunks:
        logger.error("No text chunks generated.")
    return text_chunks


def save_vectors_to_json(text_chunks, embeddings_model, base_file_name):
    if not text_chunks:
        logger.error("Text chunks are empty. No vectors will be saved.")
        return
    vectors = embeddings_model.encode(text_chunks)
    if vectors.size == 0:
        logger.error("No vectors generated from text chunks.")
        return
    file_name = f"{base_file_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(file_name, 'w', encoding='utf-8') as file:
        json.dump(vectors.tolist(), file, ensure_ascii=False, indent=4)
    logger.info(f"Vectors saved to {file_name}")

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))
