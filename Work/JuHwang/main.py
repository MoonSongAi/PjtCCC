from OCR_visualization import plt_imshow, putText, detect_text, load_terms, load_special_characters, combine_boxes_for_specific_words_1, combine_boxes_for_specific_words_2, combine_boxes_for_specific_words_3, draw_bounding_box, correct_and_visualize


# 예제 사용법 (이 부분은 실제 코드에 맞게 조정 필요)
correction_dict_1 = load_terms('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_주의어.csv')
correction_dict_2 = load_terms('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_붙여쓰기.csv')
# correction_dict_3 = load_terms('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_띄어쓰기.csv')
correction_dict_3 = load_terms('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_띄어쓰기_보류.csv')
special_chars_2 = load_special_characters('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_붙여쓰기_특정문자.csv')
special_chars_3 = load_special_characters('C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\맞춤법\\맞춤법용어집_띄어쓰기_특정문자.csv')
# image_path = '../../OCR_test/test16.png'
image_path = 'C:\\Users\\bluecom010\\Desktop\\test_final_2.png'
# image_path = 'C:\\Users\\bluecom010\\Desktop\\지의\\24_03_22_최종프로젝트\\OCR_test\\test_final_2.png'
texts = detect_text(image_path)  # detect_text 함수로부터 얻은 텍스트
# texts_rev1 = combine_boxes_for_specific_words_1(texts, ["유통", "기한"])
# combined_texts = combine_boxes_for_specific_words_2(texts_rev1)

# texts_rev1 = combine_boxes_for_specific_words_3(texts, special_chars_3) ##g, mg이런 단위 들이 앞에 있는 단어와 5미만일 떄 합쳐서 bbox
texts_rev2 = combine_boxes_for_specific_words_2(texts, special_chars_2) ##special_chars에 ['(',')'] 넣으면 괄호 앞단어 뒷단어가 좌표5이상 띄어져 있을 때 함꼐 bbox쳐짐
combined_texts = combine_boxes_for_specific_words_1(texts_rev2, ["유통", "기한"]) ##합쳐서 bbox치고 싶은 단어를 list로 받아서 bbox쳐지도록 함수화 시킴 ex)'유통' 다음 다음이 '기한'일 경우 함꼐 bbox쳐짐 
correct_and_visualize(image_path, combined_texts, correction_dict_1, correction_dict_2, correction_dict_3)
