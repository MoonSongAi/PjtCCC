from google.cloud import vision
import io
import pandas as pd
from PIL import Image, ImageDraw
import os

def read_words_from_csv(csv_file_path):
    words_df = pd.read_csv(csv_file_path)
    return words_df['Word'].tolist()

def combine_boxes_for_specific_words_1(texts, word_sequence):
    combined_texts = []
    skip_next = False
    for i in range(len(texts) - 1):
        if skip_next:
            skip_next = False
            continue
        text = texts[i]
        next_text = texts[i + 1]
        description = text.description
        next_description = next_text.description
        if description == word_sequence[0] and next_description == word_sequence[1]:
            combined_box = {
                "description": description + next_description,
                "boundingPoly": {
                    "vertices": [
                        {'x': min(text.bounding_poly.vertices[0].x, next_text.bounding_poly.vertices[0].x),
                         'y': min(text.bounding_poly.vertices[0].y, next_text.bounding_poly.vertices[0].y)},
                        {'x': max(text.bounding_poly.vertices[1].x, next_text.bounding_poly.vertices[1].x),
                         'y': min(text.bounding_poly.vertices[1].y, next_text.bounding_poly.vertices[1].y)},
                        {'x': max(text.bounding_poly.vertices[2].x, next_text.bounding_poly.vertices[2].x),
                         'y': max(text.bounding_poly.vertices[2].y, next_text.bounding_poly.vertices[2].y)},
                        {'x': min(text.bounding_poly.vertices[3].x, next_text.bounding_poly.vertices[3].x),
                         'y': max(text.bounding_poly.vertices[3].y, next_text.bounding_poly.vertices[3].y)}
                    ]
                }
            }
            combined_texts.append(combined_box)
            skip_next = True
        else:
            combined_texts.append({
                "description": text.description,
                "boundingPoly": {
                    "vertices": [{'x': vertex.x, 'y': vertex.y} for vertex in text.bounding_poly.vertices]
                }
            })

    if not skip_next and texts:
        last_text = texts[-1]
        combined_texts.append({
            "description": last_text.description,
            "boundingPoly": {
                "vertices": [{'x': vertex.x, 'y': vertex.y} for vertex in last_text.bounding_poly.vertices]
            }
        })

    return combined_texts

def detect_text(image_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "C:\\feisty-enigma-418609-e0cd3a1f9381.json"

    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    combined_texts = combine_boxes_for_specific_words_1(texts, ["프랑스", "산"])

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
            draw.rectangle([start, end], outline='red', width=2)

    if image.mode == 'RGBA':
        image = image.convert('RGB')

    image.save(save_path)
    image.show()

# 메인 실행 함수
def main():
    csv_file_path = "C:\\PjtCCC\\Work\\JGChoi\\words3.csv"
    image_path = "C:\\PjtCCC\\Work\\JGChoi\\위슬로_섭취량.png"
    save_path = "C:\\PjtCCC\\Work\\JGChoi\\path_to_save_image.jpg"

    words_to_find = read_words_from_csv(csv_file_path)
    detections = detect_text(image_path)
    draw_boxes(image_path, detections, words_to_find, save_path)

if __name__ == "__main__":
    main()
