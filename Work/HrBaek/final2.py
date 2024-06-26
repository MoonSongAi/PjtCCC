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
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Your Json Path' #json으로 발급받은 API키 입력 
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

def combine_boxes_for_specific_words_2_rev1(texts, special_chars):
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

def combine_boxes_for_specific_words_JG(texts, sequences):
    combined_texts = []
    i = 0
    while i < len(texts):
        matched = False
        for sequence in sequences:
            if i + len(sequence) <= len(texts) and all(texts[i + j]['description'] == sequence[j] for j in range(len(sequence))):
                combined_description = ''.join(texts[i + j]['description'] for j in range(len(sequence)))
                vertices = [
                    {'x': min(texts[i + j]['bounding_poly']['vertices'][0]['x'] for j in range(len(sequence))),
                     'y': min(texts[i + j]['bounding_poly']['vertices'][0]['y'] for j in range(len(sequence)))},
                    {'x': max(texts[i + j]['bounding_poly']['vertices'][2]['x'] for j in range(len(sequence))),
                     'y': max(texts[i + j]['bounding_poly']['vertices'][2]['y'] for j in range(len(sequence)))}
                ]
                combined_texts.append({
                    "description": combined_description,
                    "bounding_poly": {"vertices": [
                        {'x': vertices[0]['x'], 'y': vertices[0]['y']},
                        {'x': vertices[1]['x'], 'y': vertices[0]['y']},
                        {'x': vertices[1]['x'], 'y': vertices[1]['y']},
                        {'x': vertices[0]['x'], 'y': vertices[1]['y']}
                    ]}
                })
                i += len(sequence) - 1
                matched = True
                break
        if not matched:
            combined_texts.append({
                "description": texts[i]['description'],
                "bounding_poly": {
                    "vertices": [{'x': vertex['x'], 'y': vertex['y']} for vertex in texts[i]['bounding_poly']['vertices']]
                }
            })
        i += 1
    return combined_texts

def draw_bounding_box(image, vertices, color, text='', text_color=(255, 255, 255), font_size=22):
    """주어진 이미지에 바운딩 박스와 텍스트를 그립니다."""
    x1, y1 = vertices[0]
    x2, y2 = vertices[2]
    # x1, y1 = vertices[1][0]
    # x2, y2 = vertices[1][2]

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # 바운딩 박스 그리기
    if text:  # 텍스트가 제공된 경우에만 그림
        image = putText(image, text, x1, y1 - 15, color=text_color, font_size=font_size)
    return image

# def correct_and_visualize(image_path, roi_img, texts, correction_dict_1, correction_dict_2, correction_dict_3, correction_dict_4):
def correct_and_visualize(roi_img, texts, correction_dict_1, correction_dict_2, correction_dict_3):
    # print("Given image path:", image_path)
    # print("Does the file exist?", os.path.exists(image_path))

    # # 이미지를 로드하고 결과를 확인
    # img = cv2.imread(image_path)
    # if img is None:
    #     raise ValueError("Failed to load image, please check the file path.")
    # else:
    #     print("Image loaded successfully")
    # img = cv2.imread(image_path)
    # roi_img = img.copy()

    for text in texts[1:]:
        # vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        vertices = [(vertex["x"], vertex["y"]) for vertex in text["bounding_poly"]["vertices"]]
        correction_text = ''
        corrected = False

        for correction_dict, box_color, text_color, message_prefix in zip(
        [correction_dict_1, correction_dict_2, correction_dict_3], 
        [(0, 0, 255), (255, 0, 0), (50, 205, 50)],
        [(255, 0, 0), (0, 0, 255), (50, 205, 50)],
        ["주의어: ", "붙여쓰기: ", "볼드체: "]):  # 각 사전에 대한 안내 메시지를 정의
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
    # plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))
    return roi_img  

def combine_words(ocr_results, proximity_threshold=5):
    combined_results = []
    current_phrase = ""
    current_box = None
    last_word_end = None
    current_line_y_max = None

    for i, result in enumerate(ocr_results):
        if i == 0: continue
        # Get the start (x_min) and end (x_max) of the bounding box for the current word
        vertices = [(vertex.x, vertex.y) for vertex in result.bounding_poly.vertices]
        x_min = vertices[0][0]
        x_max = vertices[1][0]
        y_min = vertices[0][1]
        y_max = vertices[2][1]

        if last_word_end is None or (x_min - last_word_end) <= proximity_threshold:
            # Check if the current box is within the proximity threshold vertically
            if current_box is not None and abs(y_max - current_line_y_max) <= proximity_threshold:
                # Extend the current phrase and bounding box
                current_phrase += result.description
                current_box.vertices[1].x = vertices[1][0]
                current_box.vertices[2].x = vertices[2][0]
                current_box.vertices[2].y = max(current_box.vertices[2].y, y_max)
            else:
                # Append to the current phrase and start a new bounding box
                if current_box is not None:
                    combined_results.append((current_phrase, current_box))
                current_phrase = result.description
                current_box = result.bounding_poly
                current_line_y_max = y_max
        else:
            # Start a new phrase and bounding box
            combined_results.append((current_phrase, current_box))
            current_phrase = result.description
            current_box = result.bounding_poly
            current_line_y_max = y_max

        # Update the end position of the last word
        last_word_end = x_max

    # Don't forget to add the last phrase and box
    if current_phrase:
        combined_results.append((current_phrase, current_box))

    return combined_results

###API입력######
def detect_cropped(cropped):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'Your Json Path' #json으로 발급받은 API키 입력 
    client_options = {'api_endpoint': 'eu-vision.googleapis.com'} #Google Cloud Vision API의 엔드포인트를 설정
    client = vision.ImageAnnotatorClient(client_options=client_options)
    # client = vision.ImageAnnotatorClient()
    image = vision.Image(content=cropped)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    # cropped 이미지를 검사할 때는 간격을 더 짧게 3으로 설정
    # print("크롭 이미지 OCR결과 : ", texts)
    texts = combine_words(texts, 3)      #####지의
    # print("OCR 3combine 결과 : ", texts)
    return texts

def checkSpace(img_path, bboxes):
    image = Image.open(img_path)# 원본 이미지를 열어 PIL 이미지 객체로 변환합니다.
    roi_img = cv2.imread(img_path)# 바운딩 박스를 그리기 위해 원본 이미지를 OpenCV 이미지 객체로 불러옵니다.

    results = []# 최종 결과를 저장할 빈 리스트를 생성합니다.
    for bbox in bboxes:# 각 바운딩 박스에 대하여 반복 처리를 합니다.
        # 바운딩 박스의 좌표를 기반으로 이미지를 자를 영역을 계산합니다.
        left = min(bbox[1], key=lambda x: x[0])[0]
        top = min(bbox[1], key=lambda x: x[1])[1]
        right = max(bbox[1], key=lambda x: x[0])[0]
        bottom = max(bbox[1], key=lambda x: x[1])[1]
        
        cropped = image.crop((left, top, right, bottom))# 계산된 영역에 따라 이미지를 자릅니다.
        
        # 자른 이미지를 바이트 스트림으로 변환합니다.
        #해주는 이유: 많은 이미지 처리 API들은 이미지 파일이 아닌, 이미지의 바이트 데이터를 요구합니다. 따라서 이미지를 바이트 형식으로 변환하여 API 호출에 적합하게 만듭니다.
        img_bytes = io.BytesIO()
        cropped.save(img_bytes, format='PNG')
        img_bytes = img_bytes.getvalue() #img_bytes.getvalue()를 호출하면, 이미지의 바이트 스트림을 얻을 수 있습니다

        # 변환된 바이트 스트림을 통해 텍스트 검출 함수를 호출합니다.
        result = detect_cropped(img_bytes)
        # print('크롭 이미지 OCR결과를 3을 기준으로 combine한 결과: ', result) #[('3', vertices {y: 2}  vertices {  x: 12  y: 3}  vertices {  x: 11  y: 24}  vertices {  y: 23}), 

        # 텍스트 검출 결과의 길이를 기준으로 공백이 있는지 확인합니다.
        # 여기서는 결과가 2 이상일 때 공백이 있다고 판단합니다.
        isSpace = True if len(result) >= 2 else False

        # 공백이 없다고 판단될 경우 결과 리스트에 해당 바운딩 박스 정보를 추가합니다.
        if not isSpace:
            results.append(bbox)
            roi_img = draw_bounding_box(roi_img, bbox[1], (0, 165, 255), text='띄어쓰기', text_color=(255, 165, 0), font_size=10)

    
    # 처리가 완료된 후, 바운딩 박스가 그려진 이미지와 검사 결과 리스트를 반환합니다.
    return results, roi_img

def filter_bboxes_with_units(results, units):
    bboxes = []
    print("***** 검사 대상 *****")
    for item in results[1:]:  # 첫 번째 결과는 전체 이미지의 텍스트를 포함할 수 있음
        text_content = item.description
        if text_content in units or any(text_content.startswith(unit) for unit in units):
            continue
        for unit in units:
            if unit in text_content and text_content.index(unit) != 0:
                bbox = [text_content, [(vertex.x, vertex.y) for vertex in item.bounding_poly.vertices]]
                bboxes.append(bbox)
                print(bbox)
                break
    return bboxes


img_path = 'test_final_2.png'
texts = detect_text(img_path)
units = ['mg', '%']
bboxes=filter_bboxes_with_units(texts, units)

# 이미지를 불러옵니다.

img = cv2.imread(img_path)
roi_img = img.copy()

# 첫 번째 함수를 사용하여 이미지에 첫 번째 세트의 바운딩 박스와 텍스트를 적용합니다.
results, roi_img = checkSpace(img_path, bboxes)
print("\n***** 검사결과 *****")
print(results)

# 두 번째 함수를 사용하여 이미지에 두 번째 세트의 바운딩 박스와 텍스트를 적용합니다.
correction_dict_1 = load_terms('Work\\HrBaek\\용어집\\맞춤법용어집_주의어.csv')
correction_dict_2 = load_terms('Work\\HrBaek\\용어집\\맞춤법용어집_붙여쓰기.csv')
# correction_dict_3 = load_terms('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_띄어쓰기.csv') # 0g, 1g,...
correction_dict_3 = load_terms('Work\\HrBaek\\용어집\\맞춤법용어집_볼드체.csv')
special_chars_2 = load_special_characters('Work\\HrBaek\\용어집\\맞춤법용어집_붙여쓰기_특정문자.csv')

texts_rev1 = combine_boxes_for_specific_words_2(texts, special_chars_2) ##special_chars에 ['(',')'] 넣으면 괄호 앞단어 뒷단어가 좌표5이상 띄어져 있을 때 함꼐 bbox쳐짐
texts_rev2 = combine_boxes_for_specific_words_1(texts_rev1, ["유통", "기한"]) ##합쳐서 bbox치고 싶은 단어를 list로 받아서 bbox쳐지도록 함수화 시킴 ex)'유통' 다음 다음이 '기한'일 경우 함꼐 bbox쳐짐 
sequences = [["프랑스", "산"],["칼슘","과"],["비타민","K"],["비타민","D"],["비타민","K"],["구연","산삼","나트륨"],["해조","칼슘"],["산화","마그네슘"],
                 ["산화","아연"],["중성","지방"],["셀룰로스","칼슘"],["칼슘","혼합"],["탄산","칼슘"],["스테아린산","마그네슘"],["(","비타민K"],
                 ["건강","기능","식품","유통","전문","판매원"],["건강","기능","식품","전문","제조원"],["발효","마그네슘"],["산호","칼슘"],["유통","전문","판매원"],
                 ["안심","칼슘"],["마그네슘","디"],["열","량"],["카제인","나트륨"],["인산","나트륨"],["칼슘","흡수율"],["*","칼슘"],[",","비타민"],["마그네슘",","],["칼슘","&","마그네슘","&","비타민"],
                 ["이상적인","칼슘",":","마그네슘"],["마그네슘,","비타민"],["(","칼슘"],["[","칼슘"],["[","마그네슘"],["[","비타민"],["마리","골드","꽃","추출물"],
                 ["[비타민","A"],["[비타민","B2","]"],["[비타민","Be","]","단백질"],["[비타민","B12","]"],["[비타민","E","]"],["베타","카로틴"],[",","베타"],
                 ["]","마리"],[",","마리"],["산","망간"],["카로틴","혼합"],["(","베타"],[",","엽산"],["단백질","및"],["엽산","대사"],["스위스","산"],[",비타민","A"],[",비타민","B"],
                 [",비타민","B123g"],[",비타민","E"],["루테인","맥스"],["Max","마리"],["(","루테인"],["건","강","기능","식품","전문","제조원"],
                 ["건강","기능","식품","유통","전","문","판매원"],["황산","망간"],["[비타민","A",",베타","카로틴"],[",비타민","A"],[",비타민","B2"],
                 [",비타민","E"],[",비타민","B123g"],["(","비타민"],["구리","쳐"],["건강","기능","식품","우동","전도","판매"],["건강","기능","식품","전문","제조","인"],
                 ["D","를"],["줌","[마그네슘"],["필요","[비타민"],["정보","[칼슘"],["D","제품"]]
combined_texts = combine_boxes_for_specific_words_JG(texts_rev2, sequences)
# combined_texts = combine_boxes_for_specific_words_1(texts_rev2, ["프랑스", "산"])

roi_img=correct_and_visualize(roi_img, combined_texts, correction_dict_1, correction_dict_2, correction_dict_3)
# plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))

border_color = (102, 102, 102)  # Black color in BGR
border_thickness = 10  # Thickness of the border

# Adding a black border to each image
img_with_border = cv2.copyMakeBorder(img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)
roi_img_with_border = cv2.copyMakeBorder(roi_img, border_thickness, border_thickness, border_thickness, border_thickness, cv2.BORDER_CONSTANT, value=border_color)

# Combine the images with borders side by side
img_combined = np.hstack((img_with_border, roi_img_with_border))

# Save the combined image to a file
combined_image_path = './combined_image3.jpg'
cv2.imwrite(combined_image_path, img_combined)

text_to_check = ' '.join([getattr(t, 'description', '').replace('\n', ' ') for t in texts])
print(text_to_check)


import pandas as pd
from openai import OpenAI

# CSV 파일을 DataFrame으로 불러옵니다.
corrections_df = pd.read_csv('Work\\HrBaek\\용어집\\맞춤법용어집_주의어.csv')

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
image_path = combined_image_path  # 이미지 파일의 경로
original_image = Image.open(image_path)
original_width, original_height = original_image.size
draw = ImageDraw.Draw(original_image)
image_width, image_height = original_image.size

# 텍스트 설정
font_path = 'Work\\HrBaek\\SUITE-ttf\\SUITE-Light.ttf'   # 사용할 폰트 파일의 경로
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
output_image_path = 'result_with_text11.jpg'
new_image.save(output_image_path)