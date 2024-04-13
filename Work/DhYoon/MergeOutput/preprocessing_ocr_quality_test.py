import io
import os

from google.cloud import vision

def combine_words(ocr_results, proximity_threshold=5):
    combined_results = []
    current_phrase = ""
    current_box = None
    last_word_end = None

    for i, result in enumerate(ocr_results):
        if i == 0: continue
        # Get the start (x_min) and end (x_max) of the bounding box for the current word
        vertices = ([(vertex.x, vertex.y)
                    for vertex in result.bounding_poly.vertices])
        x_min = vertices[0][0]
        x_max = vertices[1][0]

        if last_word_end is None or \
            (x_min - last_word_end) <= proximity_threshold:
            # Append to the current phrase
            current_phrase += result.description
            # Extend the current bounding box
            if current_box is None:
                current_box = result.bounding_poly
            else:
                current_box.vertices[1].x = vertices[1][0]
                current_box.vertices[2].x = vertices[2][0]
        else:
            # Start a new phrase and bounding box
            combined_results.append((current_phrase, current_box))
            current_phrase = result.description
            current_box = result.bounding_poly
        
        # Update the end position of the last word
        last_word_end = x_max

    # Don't forget to add the last phrase and box
    if current_phrase:
        combined_results.append((current_phrase, current_box))
    
    return combined_results

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'API json' #구글api jason path 입력
 
client_options = {'api_endpoint': 'eu-vision.googleapis.com'}

def detect_text(path):
    client = vision.ImageAnnotatorClient(client_options=client_options)



    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    try:
        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        # print('Texts:')

        test = combine_words(texts, 5)
        # print([item[0] for item in test])
        # print([bbox[1] for bbox in test])
        
        return test
    except Exception as e:
        raise Exception(f"Error: {e}")

def detect_text2(path2):
    client = vision.ImageAnnotatorClient(client_options=client_options)



    with io.open(path2, 'rb') as image_file2:
        content2 = image_file2.read()

    try:
        image2 = vision.Image(content2=content2)

        response2 = client.text_detection(image2=image2)
        texts2 = response2.text_annotations
        # print('Texts:')

        test2 = combine_words(texts2, 5)
        # print([item[0] for item in test])
        # print([bbox[1] for bbox in test])
        
        return test2
    except Exception as e:
        raise Exception(f"Error: {e}")

result = detect_text("c.png")


result2 = detect_text("c_b.png")


text_to_check = ' '.join([t[0] for t in result])
print(text_to_check)

text_to_check2 = ' '.join([t[0] for t in result2])
print(text_to_check2)

from konlpy.tag import Kkma

def determine_superior_text(text_to_check, text_to_check2):
    # 형태소 분석기 초기화
    kkma = Kkma()

    # 첫 번째 텍스트의 문법적 오류 개수 계산
    errors1 = len(kkma.pos(text_to_check, flatten=False))

    # 두 번째 텍스트의 문법적 오류 개수 계산
    errors2 = len(kkma.pos(text_to_check2, flatten=False))

    # 오류가 적은 쪽을 우월한 텍스트로 결정
    if errors1 < errors2:
        return "첫 번째 텍스트가 더 우월합니다."
    elif errors1 > errors2:
        return "두 번째 텍스트가 더 우월합니다."
    else:
        return "두 텍스트의 우월함을 결정할 수 없습니다."

