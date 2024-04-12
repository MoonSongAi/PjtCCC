# text_detection_comparison.py
import io
import os
from google.cloud import vision
from konlpy.tag import Kkma

class TextDetectionAndComparison:
    def __init__(self, google_api_credentials, client_options={'api_endpoint': 'eu-vision.googleapis.com'}):
        os.environ['api'] = google_api_credentials
        self.client = vision.ImageAnnotatorClient(client_options=client_options)
        self.kkma = Kkma()
    
    @staticmethod
    def combine_words(ocr_results, proximity_threshold=5):
        combined_results = []
        current_phrase = ""
        current_box = None
        last_word_end = None

        for i, result in enumerate(ocr_results):
            if i == 0: continue
            vertices = [(vertex.x, vertex.y) for vertex in result.bounding_poly.vertices]
            x_min, x_max = vertices[0][0], vertices[1][0]

            if last_word_end is None or (x_min - last_word_end) <= proximity_threshold:
                current_phrase += result.description
                if current_box is None:
                    current_box = result.bounding_poly
                else:
                    current_box.vertices[1].x = vertices[1][0]
                    current_box.vertices[2].x = vertices[2][0]
            else:
                combined_results.append((current_phrase, current_box))
                current_phrase = result.description
                current_box = result.bounding_poly

            last_word_end = x_max

        if current_phrase:
            combined_results.append((current_phrase, current_box))
        return combined_results
    
    def detect_text(self, path):
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        try:
            image = vision.Image(content=content)
            response = self.client.text_detection(image=image)
            texts = response.text_annotations
            combined_texts = self.combine_words(texts, 5)
            return ' '.join([t[0] for t in combined_texts])
        except Exception as e:
            raise Exception(f"Error: {e}")

    def determine_superior_text(self, text_to_check, text_to_check2):
        errors1 = len(self.kkma.pos(text_to_check, flatten=False))
        errors2 = len(self.kkma.pos(text_to_check2, flatten=False))

        if errors1 < errors2:
            return "첫 번째 텍스트가 더 우월합니다."
        elif errors1 > errors2:
            return "두 번째 텍스트가 더 우월합니다."
        else:
            return "두 텍스트의 우월함을 결정할 수 없습니다."
