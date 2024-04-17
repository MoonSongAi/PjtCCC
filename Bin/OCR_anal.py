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
class specialDicForOCR:
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

    def init_load_font(self,size=20):
        font_path = './SUIT-ttf/SUIT-Light.ttf'   # 사용할 폰트 파일의 경로
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

    def combine_boxes_for_specific_words_2(self,texts):
        combined_texts = []
        i = 0

        while i < len(texts):
            # 이하 로직은 동일하며, description in [")", "("] 대신에
            # description in special_chars 조건을 사용하여 괄호 처리
            text = texts[i]
            description = text.description

            combine_with_next = False
            combine_with_prev = False

            if i > 0:
                prev_text = texts[i - 1]
                prev_right_x = prev_text.bounding_poly.vertices[1].x
                current_left_x = text.bounding_poly.vertices[0].x
                if current_left_x - prev_right_x >= 5 and description in self.special_chars_2:
                    combine_with_prev = True

            if i < len(texts) - 1:
                next_text = texts[i + 1]
                next_description = next_text.description
                current_right_x = text.bounding_poly.vertices[1].x
                next_left_x = next_text.bounding_poly.vertices[0].x
                if next_left_x - current_right_x >= 5 and description in self.special_chars_2:
                    combine_with_next = True

            # 이하 로직은 동일...
            if combine_with_next or combine_with_prev:
                # 다음 단어 또는 이전 단어와 합침
                new_description = description + " " + next_description if combine_with_next else prev_text.description + " " + description
                new_vertices = [
                    {"x": text.bounding_poly.vertices[0].x, "y": text.bounding_poly.vertices[0].y},
                    {"x": next_text.bounding_poly.vertices[1].x, "y": next_text.bounding_poly.vertices[1].y},
                    {"x": next_text.bounding_poly.vertices[2].x, "y": next_text.bounding_poly.vertices[2].y},
                    {"x": text.bounding_poly.vertices[3].x, "y": text.bounding_poly.vertices[3].y}
                ] if combine_with_next else [
                    {"x": prev_text.bounding_poly.vertices[0].x, "y": prev_text.bounding_poly.vertices[0].y},
                    {"x": text.bounding_poly.vertices[1].x, "y": text.bounding_poly.vertices[1].y},
                    {"x": text.bounding_poly.vertices[2].x, "y": text.bounding_poly.vertices[2].y},
                    {"x": prev_text.bounding_poly.vertices[3].x, "y": prev_text.bounding_poly.vertices[3].y}
                ]
                combined_box = {
                    "description": new_description,
                    "bounding_poly": {"vertices": new_vertices}
                }
                combined_texts.append(combined_box)
                i += 2 if combine_with_next else 1
            else:
                # 기존 박스를 유지하는 경우도 딕셔너리 형태로 변환하여 추가
                unchanged_box = {
                    "description": description,
                    "bounding_poly": {
                        "vertices": [
                            {"x": vertex.x, "y": vertex.y} for vertex in text.bounding_poly.vertices
                        ]
                    }
                }
                combined_texts.append(unchanged_box)
                i += 1

        return combined_texts

    def combine_boxes_for_specific_words_1(self, texts, word_sequence):
        combined_texts = []
        skip_next = False

        for i, text in enumerate(texts[:-1]):  # 마지막 요소 바로 전까지 순회
            if skip_next:
                skip_next = False
                continue

            # 딕셔너리 키 접근 방식으로 수정
            description = text['description']
            next_description = texts[i + 1]['description']

            if description == word_sequence[0] and next_description == word_sequence[1]:
                # 바운딩 박스를 합침
                combined_box = {
                    "locale": text.get('locale', ''),  # .get() 메서드를 사용해 안전하게 키에 접근
                    "description": description + next_description,  # 띄어쓰기를 추가
                    "bounding_poly": {
                        "vertices": [
                            text['bounding_poly']['vertices'][0],
                            texts[i + 1]['bounding_poly']['vertices'][1],
                            texts[i + 1]['bounding_poly']['vertices'][2],
                            text['bounding_poly']['vertices'][3]
                        ]
                    }
                }
                combined_texts.append(combined_box)
                skip_next = True  # 다음 단어는 이미 처리됐으므로 건너뛰기
            else:
                combined_texts.append(text)

        if not skip_next:  # 마지막 요소 처리
            combined_texts.append(texts[-1])

        return combined_texts
    
    def draw_bounding_box(self,image, vertices, color, text='', text_color=(255, 255, 255), font_size=22):
        """주어진 이미지에 바운딩 박스와 텍스트를 그립니다."""
        x1, y1 = vertices[0]
        x2, y2 = vertices[2]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 바운딩 박스 그리기
        if text:  # 텍스트가 제공된 경우에만 그림
            image = self.putText(image, text, x1, y1 - 15, color=text_color, font_size=font_size)
        return image

    def putText(self, image, text, x, y, color=(0, 255, 0), font_size=22):
        if type(image) == np.ndarray:
            color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(color_coverted)
    
        if platform.system() == 'Darwin':   #macOD에서
            font = 'AppleGothic.ttf'
        elif platform.system() == 'Windows':#윈도우에서 맑음
            font = 'malgun.ttf'
        else:                               #OS에서
            font = 'NanumGothic.ttf'
            
        image_font = ImageFont.truetype(font, font_size) #이미지에 텍스트를 그릴 때 사용
        font = ImageFont.load_default() # 시스템의 기본 폰트를 로드하여 폰트 객체를 생성
        draw = ImageDraw.Draw(image)
    
        draw.text((x, y), text, font=image_font, fill=color) # 텍스트를 실제 이미지 위에 그리는데 사용
        #text : 이미지에 그릴 문자열
        #(x, y): 텍스트를 그리기 시작할 위치의 좌표입니다. x는 가로 위치, y는 세로 위치를 나타냅니다. 이 좌표는 텍스트의 왼쪽 상단 모서리를 기준으로 합니다
        
        numpy_image = np.array(image)
        opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    
        return opencv_image

    
    def correct_and_visualize(self, image, texts):
        # img = cv2.imread(image_path)
        # bytes 데이터를 NumPy 배열로 변환
        nparr = np.frombuffer(image, np.uint8)
        # NumPy 배열을 이미지로 디코딩
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        roi_img = img.copy()

        for text in texts[1:]:
            # vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            vertices = [(vertex["x"], vertex["y"]) for vertex in text["bounding_poly"]["vertices"]]
            correction_text = ''
            corrected = False

            for correction_dict, box_color, text_color, message_prefix in zip(
            [self.correction_dict_1, self.correction_dict_2, self.correction_dict_3], 
            [(0, 0, 255), (255, 0, 0), (0, 165, 255)],[(255, 0, 0), (0, 0, 255),(255, 165, 0)],
            ["주의어: ", "붙여쓰기: ", "띄어쓰기: "]):  # 각 사전에 대한 안내 메시지를 정의
                for wrong, correct in correction_dict.items():
                    if wrong in text["description"]:
                    # if wrong in text.description:
                        correction_text = f"{message_prefix} {correct}"  # 안내 메시지를 포함한 교정 텍스트
                        # correction_text = f""  # 안내 메시지를 포함한 교정 텍스트
                        # correction_text = f"{message_prefix}Correct: {correct}"  # 안내 메시지를 포함한 교정 텍스트
                        roi_img = self.draw_bounding_box(roi_img, vertices, box_color, correction_text, text_color, font_size=10)
                        corrected = True
                        break
                if corrected:
                    break


            # Assume img and roi_img are numpy ndarrays representing your images
        border_color = (102, 102, 102)  # Black color in BGR
        border_thickness = 10  # Thickness of the border

        # Adding a black border to each image
        img_with_border = cv2.copyMakeBorder(img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
        roi_img_with_border = cv2.copyMakeBorder(roi_img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)

        # Combine the images with borders side by side
        img_combined = np.hstack((img_with_border, roi_img_with_border))

        # Save the combined image to a file
        # combined_image_path = 'Work\\HrBaek\\combined_image.jpg'
        # cv2.imwrite(combined_image_path, img_combined)

        # Return the path where the combined image is saved
        # return combined_image_path
        return img_combined

    def wrap_text(self, text, max_width):
        # 빈 이미지 생성
        dummy_image = Image.new('RGB', (100, 100))
        draw = ImageDraw.Draw(dummy_image)
        # 먼저 단일 문자의 너비를 측정합니다.
        # single_char_width = draw.te('A', font=self.font)[0]
        text_width = draw.textlength('A', font=self.font)
        # 이미지의 너비에 대해 몇 개의 'A' 문자가 맞는지 계산합니다.
        # 이를 사용하여 대략적인 문자 수 제한을 설정합니다.
        approx_chars_per_line = max_width // text_width
        wrapped_lines = textwrap.wrap(text, width=approx_chars_per_line)
        return wrapped_lines



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
