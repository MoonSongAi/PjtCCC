# langchain_integration.py
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
# from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

import os 

def is_vector_db(index_name="VECTOR_DB_INDEX"):
     # 파일 이름을 경로와 함께 구성합니다. 현재 디렉토리를 기준으로 합니다.
    # 파일 확장자는 일반적으로 '.index'입니다.
    index_path = f"./{index_name}"

    # 파일이 존재하는지 체크합니다.
    if os.path.exists(index_path):
        print(f"FAISS 인덱스 '{index_name}'가 존재합니다.")
        return True
    else:
        print(f"FAISS 인덱스 '{index_name}'가 존재하지 않습니다.")
        return False

def load_chain(index_name,device_option,openai_api_key,model_name):
    # 로컬에 저장된 데이터베이스를 불러와 new_db 변수에 할당합니다.
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': device_option},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  

    new_db = FAISS.load_local(index_name, embeddings,
                          allow_dangerous_deserialization=True)

    conversation_chain = get_conversation_chain(new_db,openai_api_key,model_name) 

    return conversation_chain

# Langchain 설정 및 초기화 함수
def setup_langchain(st , tab, uploaded_files,chunk_size,chunk_overlap,device_option,openai_api_key,model_name):

    files_text = get_text(uploaded_files)
    
    display_document_page(tab, files_text)

    text_chunks = get_text_chunks(files_text, chunk_size, chunk_overlap)
    vetorestore = get_vectorstore(text_chunks,device_option)

    vetorestore.save_local("VECTOR_DB_INDEX")
    
    conversation_chain = get_conversation_chain(vetorestore,openai_api_key,model_name) 

    return conversation_chain

def display_document_page(tab, documents):   
    first_source = '' 
    for i in range(len(documents)):
        doc = str(documents[i])
        # print('='*100)
        # print(doc)
        # print('='*100)

        start = doc.find("page_content=") + len("page_content=") +2 
        end = doc.find("metadata=") -2
        extracted_content = doc[start:end]
        # st.write(extracted_content)
        extracted_content = extracted_content.replace('\\n','<br>')

        # metadata 시작 부분을 찾습니다
        start = doc.find("metadata=") + len("metadata=")
        # metadata 종료 부분을 찾습니다 (이 경우, 마지막 괄호 전까지)
        end = doc.rfind("}")+ 1
        # metadata 문자열을 추출합니다
        metadata_str = doc[start:end]
        # metadata 문자열을 딕셔너리로 변환합니다
        # 주의: 실제 코드에서는 더 견고한 파싱 방법을 사용해야 할 수도 있습니다.
        #    print(f'start={start},end ={end} ,{metadata_str}')
        import ast
        metadata = ast.literal_eval(metadata_str)
        # metadata에서 'source'와 'page' 정보를 추출합니다
        source = metadata['source']
        page = metadata['page']
        if first_source != source:
            tab.subheader('source:'+ source)
            first_source = source
        tab.markdown(extracted_content,unsafe_allow_html=True)
        tab.caption('page No:'+ str(page))

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text, chunk_size=900, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks, device_option ='cpu'):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': device_option},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key,model_name='gpt-3.5-turbo'):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = model_name ,temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain