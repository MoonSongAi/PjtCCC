from google.cloud import vision
import io
import pandas as pd
from PIL import Image, ImageDraw
import os

## CSV파일 읽어내기
def read_words_from_csv(csv_file_path):
    words_df = pd.read_csv(csv_file_path)
    return words_df['Word'].tolist()

## 특정단어 찾아 박스그리기 
def combine_boxes_for_specific_words_1(texts, sequences):
    combined_texts = []
    i = 0
    while i < len(texts):
        matched = False
        for sequence in sequences:
            if i + len(sequence) <= len(texts) and all(texts[i + j].description == sequence[j] for j in range(len(sequence))):
                combined_description = ''.join(texts[i + j].description for j in range(len(sequence)))
                vertices = [
                    {'x': min(texts[i + j].bounding_poly.vertices[0].x for j in range(len(sequence))),
                     'y': min(texts[i + j].bounding_poly.vertices[0].y for j in range(len(sequence)))},
                    {'x': max(texts[i + j].bounding_poly.vertices[2].x for j in range(len(sequence))),
                     'y': max(texts[i + j].bounding_poly.vertices[2].y for j in range(len(sequence)))}
                ]
                combined_texts.append({
                    "description": combined_description,
                    "boundingPoly": {"vertices": [
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
                "description": texts[i].description,
                "boundingPoly": {
                    "vertices": [{'x': vertex.x, 'y': vertex.y} for vertex in texts[i].bounding_poly.vertices]
                }
            })
        i += 1

    return combined_texts

## 구글OCR후 특정단어들 합치기 
def detect_text(image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:\\feisty-enigma-418609-e0cd3a1f9381.json"
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations[1:]

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
                
    combined_texts = combine_boxes_for_specific_words_1(texts, sequences)

    print("OCR Results:")
    for detection in combined_texts:
        vertices = detection['boundingPoly']['vertices']
        print(f"Text: {detection['description']}, Vertices: {vertices}")

    return combined_texts

def draw_boxes(image_path, detections, words_to_find, save_path):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for detection in detections:
        description = detection['description']
        if description in words_to_find:
            vertices = detection['boundingPoly']['vertices']
            if len(vertices) < 4:
                continue
            start = (vertices[0]['x'], vertices[0]['y'])
            end = (vertices[2]['x'], vertices[2]['y'])
            draw.rectangle([start, end], outline='green', width=2)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save(save_path)
    image.show()

# 메인 실행 함수
def main():
    csv_file_path = "C:\\PjtCCC\\Work\\JGChoi\\words3.csv"
    image_path = "C:\PjtCCC\Work\JGChoi\안심_1.png"
    save_path = "C:\\PjtCCC\\Work\\JGChoi\\path_to_save_image.jpg"

    words_to_find = read_words_from_csv(csv_file_path)
    detections = detect_text(image_path)
    draw_boxes(image_path, detections, words_to_find, save_path)

if __name__ == "__main__":
    main()