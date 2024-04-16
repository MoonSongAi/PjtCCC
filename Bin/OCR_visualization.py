import io
import os
import cv2
import csv
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from openai import OpenAI
from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont
import textwrap
class specialDictionary:
    def __init__(self):
        self.correction_dict_1 = {}
        self.correction_dict_2 = {}
        self.correction_dict_3 = {}
        self.special_chars_2 = {}
        self.special_chars_3 = {}
        self.client = None
        self.font = None
        self.load_special_dic()
        self.init_google_vision()
        self.init_load_font(20)
    
    
    def load_terms(self, csv_path):
        # """맞춤법 교정 용어집을 로드합니다."""
        terms_df = pd.read_csv(csv_path)
        correction_dict = dict(zip(terms_df['잘못된 표현'], terms_df['올바른 표현']))
        return correction_dict

    def load_special_characters(self, csv_path):
        characters = []
                                       # utf-8로 했더니 ['\ufeff(', ')']  이렇게 출력되어서 수정함 
        with open(csv_path, newline='', encoding='utf-8-sig') as csvfile: 
            reader = csv.reader(csvfile)
            for row in reader:
                characters.extend(row)  # 각 행의 항목들을 리스트에 추가
        return characters

    def init_load_font(self,size=10):
        font_path = 'SUITE-ttf\SUITE-Light.ttf'   # 사용할 폰트 파일의 경로
        font_size = size  # 폰트 크기 설정
        self.font = ImageFont.truetype(font_path, font_size)


    def load_special_dic(self):
        # 예제 사용법 (이 부분은 실제 코드에 맞게 조정 필요)
        self.correction_dict_1 = self.load_terms('./SpecialDic/맞춤법용어집_주의어.csv')
        self.correction_dict_2 = self.load_terms('./SpecialDic/맞춤법용어집_붙여쓰기.csv')
        self.correction_dict_3 = self.load_terms('./SpecialDic/맞춤법용어집_띄어쓰기_보류.csv')
        self.special_chars_2 = self.load_special_characters('./SpecialDic/맞춤법용어집_붙여쓰기_특정문자.csv')
        self.special_chars_3 = self.load_special_characters('./SpecialDic/맞춤법용어집_띄어쓰기_특정문자.csv')

    def init_google_vision(self):
        """이미지 파일에서 텍스트를 감지합니다."""
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'C:\\Users\\user\\Desktop\\myccc-420108-7f52a40950c8.json' #json으로 발급받은 API키 입력 
        client_options = {'api_endpoint': 'eu-vision.googleapis.com'} #Google Cloud Vision API의 엔드포인트를 설정
        self.client = vision.ImageAnnotatorClient(client_options=client_options)#Google Cloud Vision API의 ImageAnnotatorClient 인스턴스를 생성합니다. 

    ###API입력######
    def detect_text(self, content):
        image = vision.Image(content=content)
        response = self.client.text_detection(image=image)
        texts = response.text_annotations
        return texts

    def call_GPT(self,text_to_check):
        originals = self.correction_dict_1.keys()
        replacements = self.correction_dict_1.values()

        # GPT 모델에게 수정 요청
        openai_api_key = st.secrets["OpenAI_Key"]
        client = OpenAI(api_key=openai_api_key)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "너는 문장을 수정해주는 맞춤법 검사기이자 문장 검수자야"
                },
                {
                    "role": "user",
                    "content": f"수정할 문장{text_to_check}, csv로 인해 고친 사항은 맞춤법 검사 수정에서 제외해줘."
                }
                ,{
                    "role":"assistant",
                    "content": f"다음 문장에서 주어진 단어들을 대체해줘. 공식적인 말투로 말해줘:\n{text_to_check}\n\n교체 규칙:\n" + "\n".join([f"{o} -> {r}" for o, r in zip(originals, replacements)])
                }
                ,{
                    "role":"assistant",
                    "content": "너는 한국어 텍스트의 맞춤법을 검사하고 수정 제안을 하는 역할이야. 텍스트를 분석하여 맞춤법 오류가 있는 경우 모든 부분의 정확한 수정 제안을 제공해줘. 공식적인 말투로 말해줘."
                }        
                ,{
                    "role":"assistant",
                    "content": "수정을 거친 총 문장으로 시작해서 2개의 문단으로 작성해 줘. 이때 첫번째는 '수정문장:'으로 시작하고 수정된 내용을, 두번째는 '변경사항:'으로 시작하고 변경된 내용을 작성해줘"
                }    
                ]
            )

        correction = response.choices[0].message.content

        return correction
