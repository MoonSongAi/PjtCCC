
import io
import os
import cv2
import csv
import platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from google.cloud import vision
from PIL import Image, ImageDraw, ImageFont

def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list: #img가 리스트일 때
        if type(title) == list: #title이 리스트일 때
            titles = title # 이는 이미 각 이미지에 대응하는 제목이 준비되어 있다는 뜻이므로, titles에 title 리스트를 그대로 할당
        else: #title이 리스트가 아닌 단일제목만 제공 된 경우, 모든 이미지에 같은 제목 사용 
            titles = [] #초기화
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2: # img.shape이 3차원 미만인 경우 
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg) #(행, 열, 서브플롯의 위치 인덱스)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([]) #x축과 y축의 눈금을 제거
 
        plt.show()
    else: #img가 리스트가 아닐 때 다닐 이미지로 간주하고 처리
        if len(img.shape) < 3: #img.shape가 3차원 미만인 경우 
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def putText(image, text, x, y, color=(0, 255, 0), font_size=22):
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

###API입력######
def detect_text(path):
    """이미지 파일에서 텍스트를 감지합니다."""
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Your json' #json으로 발급받은 API키 입력 
    client_options = {'api_endpoint': 'eu-vision.googleapis.com'} #Google Cloud Vision API의 엔드포인트를 설정
    client = vision.ImageAnnotatorClient(client_options=client_options)#Google Cloud Vision API의 ImageAnnotatorClient 인스턴스를 생성합니다. 
    # client = vision.ImageAnnotatorClient()
    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts

def load_terms(csv_path):
    """맞춤법 교정 용어집을 로드합니다."""
    terms_df = pd.read_csv(csv_path)
    correction_dict = dict(zip(terms_df['잘못된 표현'], terms_df['올바른 표현']))
    return correction_dict

def load_special_characters(csv_path):
    """
    .csv 파일에서 특정 문자 목록을 로드합니다.

    Parameters:
    csv_path (str): 특정 문자 목록이 저장된 .csv 파일의 경로.

    Returns:
    list: 파일에서 로드된 특정 문자 목록.
    """
    characters = []
    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile: ### utf-8로 했더니 ['\ufeff(', ')']  이렇게 출력되어서 수정함 
        reader = csv.reader(csvfile)
        for row in reader:
            characters.extend(row)  # 각 행의 항목들을 리스트에 추가
    return characters

def combine_boxes_for_specific_words_1(texts, word_sequence):
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

def combine_boxes_for_specific_words_2(texts, special_chars):
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
            if current_left_x - prev_right_x >= 5 and description in special_chars:
                combine_with_prev = True

        if i < len(texts) - 1:
            next_text = texts[i + 1]
            next_description = next_text.description
            current_right_x = text.bounding_poly.vertices[1].x
            next_left_x = next_text.bounding_poly.vertices[0].x
            if next_left_x - current_right_x >= 5 and description in special_chars:
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

def combine_boxes_for_specific_words_3(texts, special_chars):
    combined_texts = []
    i = 0

    while i < len(texts):
        text = texts[i]
        description = text.description

        combine_with_prev = False

        if i > 0:
            prev_text = texts[i - 1]
            prev_right_x = prev_text.bounding_poly.vertices[1].x
            current_left_x = text.bounding_poly.vertices[0].x
            # 현재 단어와 이전 단어 사이의 거리가 5 미만이고, 현재 단어가 특정 문자 리스트에 속하는 경우에만 합칩니다.
            if current_left_x - prev_right_x < 5 and description in special_chars:
                combine_with_prev = True

        if combine_with_prev:
            # 이전 단어와 현재 단어를 합침
            combined_box = {
                "description": prev_text.description + " " + description,
                "bounding_poly": {
                    "vertices": [
                        {"x": prev_text.bounding_poly.vertices[0].x, "y": prev_text.bounding_poly.vertices[0].y},
                        {"x": text.bounding_poly.vertices[1].x, "y": text.bounding_poly.vertices[1].y},
                        {"x": text.bounding_poly.vertices[2].x, "y": text.bounding_poly.vertices[2].y},
                        {"x": prev_text.bounding_poly.vertices[3].x, "y": prev_text.bounding_poly.vertices[3].y}
                    ]
                }
            }
            # 마지막에 추가된 합쳐진 박스를 제거하고 새로운 박스를 추가합니다.
            if combined_texts and combined_texts[-1]['description'] == prev_text.description:
                combined_texts.pop()
            combined_texts.append(combined_box)
        else:
            # 이전 단어와 합쳐지지 않는 경우, 현재 단어를 그대로 유지합니다.
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

def draw_bounding_box(image, vertices, color, text='', text_color=(255, 255, 255), font_size=22):
    """주어진 이미지에 바운딩 박스와 텍스트를 그립니다."""
    x1, y1 = vertices[0]
    x2, y2 = vertices[2]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 바운딩 박스 그리기
    if text:  # 텍스트가 제공된 경우에만 그림
        image = putText(image, text, x1, y1 - 15, color=text_color, font_size=font_size)
    return image

def correct_and_visualize(image_path, texts, correction_dict_1, correction_dict_2, correction_dict_3):
    img = cv2.imread(image_path)
    roi_img = img.copy()

    for text in texts[1:]:
        # vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        vertices = [(vertex["x"], vertex["y"]) for vertex in text["bounding_poly"]["vertices"]]
        correction_text = ''
        corrected = False

        for correction_dict, box_color, text_color, message_prefix in zip(
        [correction_dict_1, correction_dict_2, correction_dict_3], 
        [(0, 0, 255), (255, 0, 0), (0, 165, 255)],[(255, 0, 0), (0, 0, 255),(255, 165, 0)],
        ["주의어: ", "붙여쓰기: ", "띄어쓰기: "]):  # 각 사전에 대한 안내 메시지를 정의
            for wrong, correct in correction_dict.items():
                if wrong in text["description"]:
                # if wrong in text.description:
                    correction_text = f"{message_prefix} {correct}"  # 안내 메시지를 포함한 교정 텍스트
                    # correction_text = f""  # 안내 메시지를 포함한 교정 텍스트
                    # correction_text = f"{message_prefix}Correct: {correct}"  # 안내 메시지를 포함한 교정 텍스트
                    roi_img = draw_bounding_box(roi_img, vertices, box_color, correction_text, text_color, font_size=10)
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
    combined_image_path = 'Work\\HrBaek\\combined_image.jpg'
    cv2.imwrite(combined_image_path, img_combined)

    # Return the path where the combined image is saved
    return combined_image_path


# In[51]:


# 예제 사용법 (이 부분은 실제 코드에 맞게 조정 필요)
correction_dict_1 = load_terms('Work\\HrBaek\\맞춤법용어집_주의어.csv')
correction_dict_2 = load_terms('Work\\HrBaek\\맞춤법용어집_붙여쓰기.csv')
# correction_dict_3 = load_terms('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_띄어쓰기.csv')
correction_dict_3 = load_terms('Work\\HrBaek\\맞춤법용어집_띄어쓰기_보류.csv')
special_chars_2 = load_special_characters('Work\\HrBaek\\맞춤법용어집_붙여쓰기_특정문자.csv')
special_chars_3 = load_special_characters('Work\\HrBaek\\맞춤법용어집_띄어쓰기_특정문자.csv')
# image_path = '../../OCR_test/test16.png'
image_path = 'Work\\HrBaek\\test_final_2.png'
# image_path = 'C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\OCR_test\\test_final_2.png'
texts = detect_text(image_path)  # detect_text 함수로부터 얻은 텍스트
# texts_rev1 = combine_boxes_for_specific_words_1(texts, ["유통", "기한"])
# combined_texts = combine_boxes_for_specific_words_2(texts_rev1)

# texts_rev1 = combine_boxes_for_specific_words_3(texts, special_chars_3) ##g, mg이런 단위 들이 앞에 있는 단어와 5미만일 떄 합쳐서 bbox
texts_rev2 = combine_boxes_for_specific_words_2(texts, special_chars_2) ##special_chars에 ['(',')'] 넣으면 괄호 앞단어 뒷단어가 좌표5이상 띄어져 있을 때 함꼐 bbox쳐짐
combined_texts = combine_boxes_for_specific_words_1(texts_rev2, ["유통", "기한"]) ##합쳐서 bbox치고 싶은 단어를 list로 받아서 bbox쳐지도록 함수화 시킴 ex)'유통' 다음 다음이 '기한'일 경우 함꼐 bbox쳐짐 
correct_and_visualize(image_path, combined_texts, correction_dict_1, correction_dict_2, correction_dict_3)


# In[52]:


text_to_check = ' '.join([getattr(t, 'description', '').replace('\n', ' ') for t in texts])
print(text_to_check)


# In[53]:


import pandas as pd
from openai import OpenAI

# CSV 파일을 DataFrame으로 불러옵니다.
corrections_df = pd.read_csv('Work\\HrBaek\\맞춤법용어집_주의어.csv')

# 'original'과 'replacement' 열을 각각 리스트로 변환합니다.
originals = corrections_df['잘못된 표현'].tolist()
replacements = corrections_df['올바른 표현'].tolist()

# GPT 모델에게 수정 요청
client = OpenAI(api_key='Your API Key')


# text_to_check = ' '.join([t[0] for t in result])

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
print(correction)


from PIL import Image, ImageDraw, ImageFont
import textwrap

def wrap_text(text, font, max_width):
    # 먼저 단일 문자의 너비를 측정합니다.
    single_char_width = draw.textsize('A', font=font)[0]
    # 이미지의 너비에 대해 몇 개의 'A' 문자가 맞는지 계산합니다.
    # 이를 사용하여 대략적인 문자 수 제한을 설정합니다.
    approx_chars_per_line = max_width // single_char_width
    wrapped_lines = textwrap.wrap(text, width=approx_chars_per_line)
    return wrapped_lines

# 이미지 로딩
# image_path = 'Work\\HrBaek\\combined_image.jpg'  # 이미지 파일의 경로
original_image = Image.open(combined_image_path)
original_width, original_height = original_image.size
draw = ImageDraw.Draw(original_image)
image_width, image_height = original_image.size

# 텍스트 설정
font_path = 'SUITE-ttf\SUITE-Light.ttf'   # 사용할 폰트 파일의 경로
font_size = 20  # 폰트 크기 설정
font = ImageFont.truetype(font_path, font_size)
margin = 10  # 이미지 가장자리와 텍스트 사이의 여백 설정

# 텍스트 줄바꿈
changes_index = correction.index('변경사항')
first_part = correction[:changes_index].strip()
second_part = correction[changes_index:].strip()
max_text_width = image_width - 2 * margin  # 최대 텍스트 폭 설정

# 두 부분을 각각 줄바꿈
wrapped_first_part = wrap_text(first_part, font, max_text_width)
wrapped_second_part = wrap_text(second_part, font, max_text_width)

# 두 부분을 결합
combined_text = wrapped_first_part + [''] + wrapped_second_part

# 텍스트 그리기를 위한 추가 코드는 여기에 포함시킵니다.
# ...
# 새 이미지의 높이 계산
line_height = font.getsize('텍스트')[1] + 10  # 줄 간격 포함
new_height = original_height + line_height * len(combined_text)

# 새 이미지 생성 및 원본 이미지 붙이기
new_image = Image.new('RGB', (original_width, new_height), 'white')
new_image.paste(original_image, (0, 0))

# 텍스트 그리기
draw = ImageDraw.Draw(new_image)
current_y = original_height + 10  # 원본 이미지 아래에 텍스트 시작

for line in combined_text:
    draw.text((10, current_y), line, fill='black', font=font)
    current_y += line_height  # 다음 텍스트 줄의 y 위치 설정

# 이미지 저장
output_image_path = 'Work\\HrBaek\\result_with_text.jpg'
new_image.save(output_image_path)

