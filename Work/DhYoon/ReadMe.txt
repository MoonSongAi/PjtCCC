## Miniconda3 환경에서 가상환경 만든다
- terminal bash 명령어
conda create --name langchain_st
conda activate langchain_st

## share.streamlit.io 에서 app을 등록 한다

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