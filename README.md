## Miniconda3 환경에서 가상환경 만든다
- terminal bash 명령어
conda create --name CCC
conda activate CCC

## share.streamlit.io 에서 app을 등록 한다
# 본 프로젝트에서는 google API , OpenAI API 등에 사용되는 json , pswd 등이 사용되어 
# 원격 배포는 하지 않는다. 

## 원격서버 대신 로칼에서 동작 하는 것으로 한다 
# >>>Streamlit run C:\PjtCCC\Bin\streamlit_app.py

## Git init and remote connection
# PC에 git-scm.com에서 Git Download & install
>>> git init  
>>> git remote add origin https://github.com/MoonSongIT/langchain_st.git

## 3/23 
## langchain_community.chat_models.openai.ChatOpenAI 클래스가 
## langchain-community 버전 0.0.10에서 사용 중단(Deprecated) 되었으며,
## 0.2.0 버전에서 제거될 예정임을 알립
>>> pip install -U langchain-openai
# from langchain_openai import ChatOpenAI

# >>>pip install streamlit-drawable-canvas
>>>pip install streamlit-cropper
#>>>pip install pdf2image
>>>pip install PyMuPDF
# 4/2 
>>>pip install pypdf
>>>pip install sentence-transformers
>>>pip install faiss-gpu
#4/4
>>>pip install docx2txt

