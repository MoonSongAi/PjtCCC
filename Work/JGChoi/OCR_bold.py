from google.cloud import vision
import io
import pandas as pd
from PIL import Image, ImageDraw
import requests
import base64
import os

# CSV 파일에서 단어 목록을 읽어오는 함수
def read_words_from_csv(csv_file_path):
    words_df = pd.read_csv(csv_file_path)
    return words_df['Word'].tolist()

# 특정 단어 시퀀스를 합치는 함수
def combine_boxes_for_specific_words_1(texts, word_sequence):
    combined_texts = []
    skip_next = False
    for i, text in enumerate(texts[:-1]):  # 마지막 요소 바로 전까지 순회
        if skip_next:
            skip_next = False
            continue
        description = text['description']
        next_description = texts[i + 1]['description']
        if description == word_sequence[0] and next_description == word_sequence[1]:
            # 바운딩 박스를 합침
            combined_box = {
                "description": description + next_description,
                "boundingPoly": {
                    "vertices": [
                        text['boundingPoly']['vertices'][0],
                        texts[i + 1]['boundingPoly']['vertices'][2]
                    ]
                }
            }
            combined_texts.append(combined_box)
            skip_next = True
        else:
            combined_texts.append(text)

    if not skip_next:
        combined_texts.append(texts[-1])

    return combined_texts

# 이미지에서 텍스트 감지
def detect_text(image_path, api_key):
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    encoded_image = base64.b64encode(content).decode()

    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    body = {
        "requests": [{
            "image": {"content": encoded_image},
            "features": [{"type": "TEXT_DETECTION"}]
        }]
    }

    response = requests.post(url, headers=headers, json=body)
    response.raise_for_status()
    result = response.json()

    detections = result['responses'][0].get('textAnnotations', [])
    # 특정 단어들을 합친 후의 결과를 반환
    detections = combine_boxes_for_specific_words_1(detections, ["프랑스","산"])

    # OCR 결과 출력
    print("OCR Results:")
    for detection in detections:
        print(f"Text: {detection['description']}, Vertices: {detection['boundingPoly']['vertices']}")

    return detections

def draw_boxes(image_path, detections, words_to_find, save_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 검출된 모든 텍스트에 대해 반복
    for detection in detections:
        description = detection['description']
        if description in words_to_find:  # CSV 파일의 단어 목록에 있는 단어인지 확인
            vertices = detection['boundingPoly']['vertices']
            if len(vertices) < 4:
                print("Skipping a box with insufficient vertices.")
                continue

            # 왼쪽 상단과 오른쪽 하단 좌표를 확보
            x0, y0 = vertices[0]['x'], vertices[0]['y']
            x1, y1 = vertices[2]['x'], vertices[2]['y']

            # 좌표를 정렬하여 x0 <= x1 과 y0 <= y1이 되도록 보장
            left, top = min(x0, x1), min(y0, y1)
            right, bottom = max(x0, x1), max(y0, y1)

            draw.rectangle([left, top, right, bottom], outline='red', width=2)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save(save_path)
    image.show()

# 메인 실행 함수
def main():
    csv_file_path = "C:\\PjtCCC\\Work\\JGChoi\\words3.csv"  # CSV 파일 경로
    image_path = "C:\\PjtCCC\\Work\\JGChoi\\비주얼_영양성분.png"
    save_path = "C:\\PjtCCC\\Work\\JGChoi\\path_to_save_image.jpg"  # 결과 이미지 저장 경로
    api_key = "API"

    # CSV 파일에서 단어 목록 읽기
    words_to_find = read_words_from_csv(csv_file_path)

    # 텍스트 감지 및 바운딩 박스 그리기
    detections = detect_text(image_path, api_key)
    draw_boxes(image_path, detections, words_to_find, save_path)

if __name__ == "__main__":
    main()
