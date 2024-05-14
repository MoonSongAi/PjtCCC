# pip install PyOpenGL PyOpenGL_accelerate
# pip install PyQt5

import os
import sys
import cv2
import numpy as np
from sklearn.cluster import KMeans  # add by yoon 5/3 16:40

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
from PyQt5.QtCore import QPoint, Qt 
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtGui import (
    QImage, 
    QPixmap, 
    QPainter, 
    QPen, 
    QColor,
    QCursor,
     
)
from extend_class import *
from glWidget import *
# 모듈 최상단에 전역 변수 정의
from config import faceKeywords, image_folder
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Box image rendering")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

        self.imagePaths = [None] * 6

        self.imageWindow = None  # imageWindow를 저장하기 위한 클래스 변수 추가
        self.imgLabel  = None # 기존 위젯 제거
        self.maskImgLabel = None
        self.lastWindowPos = None  # 창의 마지막 위치를 저장할 변수

        self.selLines = []
        self.interSections =[]
        self.mskEdges = None  # msk_edges를 저장할 속성
        self.qImage = None  # QImage 객체

        self.click_count = 0
        self.faceBoxes = None # 유니크한 박스 저장

    def initUI(self):
        self.centralWidget = QWidget(self)
        self.setCentralWidget(self.centralWidget)
        layout = QVBoxLayout(self.centralWidget)

        self.widthInput = QLineEdit("1.0")
        self.heightInput = QLineEdit("1.0")
        self.depthInput = QLineEdit("1.0")

        self.glWidget = GLWidget(self) 
        layout.addWidget(self.glWidget)        

        btnBatchSelect = QPushButton("이미지선택", self)
        # btnBatchSelect.clicked.connect(self.openMultipleImagesFileDialog)
        btnBatchSelect.clicked.connect(self.loadBoxFace)
        layout.addWidget(btnBatchSelect)

        # "이미지로드" 버튼 추가
        btnLoadImage = QPushButton("이미지로드", self)
        btnLoadImage.clicked.connect(self.loadImage)
        layout.addWidget(btnLoadImage)

        inputLayout = QHBoxLayout()
        sizeInputs = [
            ("Width", self.widthInput),
            ("Height", self.heightInput),
            ("Depth", self.depthInput),
        ]
        for label, widget in sizeInputs:

            inputLayout.addWidget(QLabel(label))
            inputLayout.addWidget(widget)
        layout.addLayout(inputLayout)

        self.widthInput.textChanged[str].connect(self.glWidget.updateCube)
        self.heightInput.textChanged[str].connect(self.glWidget.updateCube)
        self.depthInput.textChanged[str].connect(self.glWidget.updateCube)
    
    def resizeEvent(self, event):
        # 크기가 조정될 때 실행할 로직
        self.update_image_display()
        super().resizeEvent(event)
    

    def handle_click(self, x, y,button_clicked):
        # 이벤트를 발생시킨 객체를 가져옵니다.
        sender_label = self.sender()

        # sender_label을 확인하여 어떤 라벨에서 이벤트가 발생했는지 판별합니다.
        if sender_label == self.imgLabel:
            self.show_click_count(x, y)
            print(f"Clicked on imgLabel at: ({x}, {y}) with {button_clicked} button")
        elif sender_label == self.maskImgLabel:
            self.toggle_point_in_list(x, y)
            self.update_image_display()
            print(f"Clicked on maskImgLabel at: ({x}, {y}) with {button_clicked} button")
        else:
            print("Clicked on an unidentified label")


    def openMultipleImagesFileDialog(self):
        imagePaths, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if not imagePaths:
            return


        for imagePath in imagePaths:
            fileName = os.path.basename(imagePath)

            for keyword, index in faceKeywords.items():
                if keyword in fileName:
                    self.imagePaths[index] = imagePath
                    self.glWidget.loadTextureForFace(imagePath, index)
                    break

    def loadBoxFace(self):
        for faceName ,index in faceKeywords.items():
            filename = os.path.join(image_folder, f'{faceName}.jpg')
            self.imagePaths[index] = filename
            self.glWidget.loadTextureForFace(filename, index)

        width, depth = self.calculate_dimension_ratios(image_folder)
        self.widthInput.setText("{:.2f}".format(width))
        self.depthInput.setText("{:.2f}".format(depth))


    def calculate_dimension_ratios(self,folder_path):
        # 이미지 파일 경로
        front_face_path = os.path.join(folder_path, "앞면.jpg")
        top_face_path = os.path.join(folder_path, "상단면.jpg")
        bottom_face_path = os.path.join(folder_path, "하단면.jpg")

        # "앞면.jpg"의 크기를 조사하여 높이와 너비 결정
        with Image.open(front_face_path) as img:
            width, height = img.size
            if width > height:
                height, width = width, height  # 넓은 쪽을 높이로 설정

        # "상단면.jpg" 혹은 "하단면.jpg"를 조사하여 깊이 결정
        depth = None
        for face_path in [top_face_path, bottom_face_path]:
            try:
                with Image.open(face_path) as img:
                    face_width, face_height = img.size
                    # 너비와 같지 않은 면의 크기를 깊이로 설정
                    if face_width != width:
                        depth = face_width
                    elif face_height != width:
                        depth = face_height
                    break  # 깊이를 찾았으면 반복 중단
            except FileNotFoundError:
                continue  # 파일이 없으면 다음 파일 검사

        if depth is None:
            print("깊이를 결정할 수 없습니다.")
            return None

        # 높이를 1로 했을 때의 너비와 깊이의 비율 계산
        width_ratio = width / height
        depth_ratio = depth / height

        return width_ratio, depth_ratio
    
############################################################################
    def line_length(self, line):
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_intersection(self,line1, line2, expansion=1):
        # 선분의 좌표를 추출합니다.
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
    # 먼저 선분이 수직인지 수평인지 판단합니다.
        vertical1 = x1 == x2  # 선분1이 수직선인 경우
        horizontal1 = y1 == y2  # 선분1이 수평선인 경우
        vertical2 = x3 == x4  # 선분2가 수직선인 경우
        horizontal2 = y3 == y4  # 선분2가 수평선인 경우

    # 수직선과 수평선이 교차하는 경우에만 교차점을 계산합니다.
        if vertical1 and horizontal2:
            # 선분1이 수직이고 선분2가 수평일 때, 수평선분을 확장합니다.
            if (x1 >= min(x3, x4) - expansion and x1 <= max(x3, x4) + expansion and
                    y3 >= min(y1, y2) - expansion and y3 <= max(y1, y2) + expansion):
                return (x1, y3)
        elif horizontal1 and vertical2:
            # 선분1이 수평이고 선분2가 수직일 때, 수직선분을 확장합니다.
            if (x3 >= min(x1, x2) - expansion and x3 <= max(x1, x2) + expansion and
                    y1 >= min(y3, y4) - expansion and y1 <= max(y3, y4) + expansion):
                return (x3, y1)

        return None  # 교차점이 없거나, 수직/수평 조건에 맞지 않는 경우

    # 선들의 배열을 받아 교차점의 배열을 반환
    def get_intersections(self,lines):
        intersections = []
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                intersect = self.find_intersection(line1, line2,10)
                if intersect is not None:
                    intersections.append(intersect)
        return intersections
    
    def remove_near_duplicates(self, intersections, tolerance):
        unique_intersections = []
        for current in intersections:
            # 현재 교차점이 이미 추가된 교차점들과 너무 가까운지 확인합니다.
            if any(np.sqrt((x - current[0]) ** 2 + (y - current[1]) ** 2) < tolerance for x, y in unique_intersections):
                # 이미 유사한 교차점이 있으면, 현재 교차점을 추가하지 않습니다.
                continue
            # 유사한 교차점이 없으면, 현재 교차점을 추가합니다.
            unique_intersections.append(current)
        return unique_intersections

    def average_within_tolerance(self, points, tolerance):
        # x와 y 각각에 대한 평균값을 계산
        x_vals, y_vals = zip(*points)  # x, y 좌표 분리
        x_means = {}
        y_means = {}

        for i, x in enumerate(x_vals):
            # x 좌표 기준 근접 그룹 찾기
            close_group = [x_val for x_val in x_vals if abs(x_val - x) <= tolerance]
            x_mean = np.mean(close_group)
            x_means[x] = int(x_mean)

        for i, y in enumerate(y_vals):
            # y 좌표 기준 근접 그룹 찾기
            close_group = [y_val for y_val in y_vals if abs(y_val - y) <= tolerance]
            y_mean = np.mean(close_group)
            y_means[y] = int(y_mean)

        # 평균값을 기존 좌표에 치환
        averaged_points = [(x_means[x], y_means[y]) for x, y in points]

        return averaged_points
    
    def create_box_from_points(self, points):
        # x 좌표와 y 좌표를 분리하여 각각의 리스트를 생성합니다.
        x_coords, y_coords = zip(*points)

        # 각 리스트에서 최소값과 최대값을 찾습니다.
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        # 박스의 꼭지점 좌표를 계산합니다.
        # (좌하단, 좌상단, 우상단, 우하단)
        box_coordinates = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

        return box_coordinates
    
    def find_non_overlapping_boxes(self, points):
        boxes = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                min_x = min(points[i][0], points[j][0])
                max_x = max(points[i][0], points[j][0])
                min_y = min(points[i][1], points[j][1])
                max_y = max(points[i][1], points[j][1])
                if (max_x - min_x)  > 0 and (max_y - min_y) > 0:                    # 면적이 0보다 큰 박스만 추가
                    if ((min_x , min_y) in points) and ((max_x , min_y) in points) and \
                       ((min_x , max_y) in points) and ((max_x , max_y) in points): # 좌상, 좌하,우상,우하 좌표가 Points에 있어야 한다
                        boxes.append([(min_x, min_y), (max_x, max_y)])
                        # print(f'[({min_x},{min_y}),({max_x},{max_y})]')
        # 중복 제거
        u_boxes = []
        for box in boxes:
            if box not in u_boxes:
                u_boxes.append(box)

        # 좌상 좌표가 같은 박스들을 찾고, 그 중 가장 작은 면적의 박스를 찾습니다.
        same_top_left_boxes = {}
        for box in u_boxes:
            top_left = box[0]
            width = box[1][0] - box[0][0]
            height = box[1][1] - box[0][1]
            area = width * height
            
            if top_left in same_top_left_boxes:
                if same_top_left_boxes[top_left]['area'] > area:
                    same_top_left_boxes[top_left] = {'box': box, 'area': area}
            else:
                same_top_left_boxes[top_left] = {'box': box, 'area': area}

        # 가장 작은 박스를 찾습니다.
        smallest_boxes = [details['box'] for details in same_top_left_boxes.values()]
        return smallest_boxes
    
    def draw_boxes_on_image(self, qImg, boxes):
        # QImage를 QPixmap으로 변환
        pixmap = QPixmap.fromImage(qImg)
        
        # QPainter 객체 생성 및 시작
        painter = QPainter(pixmap)
        
        # 점선 스타일 설정
        pen = QPen(QColor(0, 0, 255), 4, Qt.DashLine)  # 감색, 두께 4, 점선 스타일
        painter.setPen(pen)
        
        # 각 박스에 대해 점선 사각형을 그림
        for box in boxes:
            left, top = box[0]
            right, bottom = box[1]
            width = right - left
            height = bottom - top
            painter.drawRect(left, top, width, height)
        
        # QPainter 사용 종료
        painter.end()
        
        # 그려진 QPixmap을 QImage로 다시 변환 (필요한 경우)
        # new_qImg = pixmap.toImage()

        # QPixmap을 반환 (이 예시에서는 QImage 대신 QPixmap 사용을 추천)
        return pixmap
    def save_boxes(self, save_path, qImg, boxes):        
        # 지정된 저장 경로가 없으면 생성합니다.
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for index , box in enumerate(boxes):
            left, top = box[0]
            right, bottom = box[1]
            width = right - left
            height = bottom - top
             # 박스 영역을 QImage로부터 추출하여 새 QPixmap을 만듦
            cropped_qimage = qImg.copy(left, top, width, height)
            cropped_pixmap = QPixmap.fromImage(cropped_qimage)

            # QPixmap을 이미지 파일로 저장
            filename = os.path.join(save_path, f'cropped_box_{index}.jpg')
            cropped_pixmap.save(filename, 'JPG')
    ############################### add by yoon 5/3 17:40
    def resize_image_for_display(self, image, max_display_size=800):
        # 이미지의 최대 크기를 조정
        height, width = image.shape[:2]
        scaling_factor = max_display_size / float(max(height, width))
        
        # 너무 크지 않다면 원본 유지
        if scaling_factor < 1:
            image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        return image
    def select_roi(self, image_path):
        # 이미지 로드
        image = cv2.imread(image_path)
        # 화면 크기에 맞게 이미지 크기 조정
        resized_image = self.resize_image_for_display(image)        
        # ROI 선택을 위한 마우스 콜백 함수
        roi = cv2.selectROI("Select ROI", resized_image)
        cv2.destroyAllWindows()
        
        # 원본 이미지 크기에 맞게 ROI 좌표 조정
        scaling_factor = image.shape[1] / float(resized_image.shape[1])
        roi = tuple([int(x * scaling_factor) for x in roi])
        
        # 원본 이미지에서 선택된 ROI의 HSV 값 추출
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        roi_hsv = hsv_image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        average_hsv = cv2.mean(roi_hsv)[:3]
        return average_hsv

    def define_hsv_range(self, hsv_value, range_width=10):
        lower_hsv = np.array([max(0, hsv_value[0] - range_width), max(0, hsv_value[1] - range_width), max(0, hsv_value[2] - range_width)])
        upper_hsv = np.array([min(179, hsv_value[0] + range_width), min(255, hsv_value[1] + range_width), min(255, hsv_value[2] + range_width)])
        return lower_hsv, upper_hsv
    
    def detect_lines_and_colors(self, image_path):
        # 이미지 로드 및 그레이스케일 변환
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough 변환을 통한 선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        
        # 선의 방향 분석 및 색상 추출
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 수평 또는 수직 선 감지
                if x1 == x2 or y1 == y2:  # 수직 선은 x1 == x2, 수평 선은 y1 == y2
                    # 선의 중간 지점 색상 추출
                    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    mid_point_color = hsv_image[(y1+y2)//2, (x1+x2)//2]
                    return mid_point_color

    # 도미넌트 색상 찾기
    def find_dominant_color(self,hsv_image, k=4):
        data = hsv_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
        return dominant_color.astype(int)

    # HSV 범위 설정
    def get_hsv_range(self, dominant_hsv, range_width=10):
        lower_hsv = np.maximum(dominant_hsv - range_width, [0, 50, 50])
        upper_hsv = np.minimum(dominant_hsv + range_width, [180, 255, 255])
        return lower_hsv, upper_hsv
    ############################### add by yoon 5/3 17:40
    def analyze_lines_and_colors(self,image_path):
        # 이미지 로드 및 그레이스케일로 변환
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Canny 엣지 검출
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough 변환을 통한 선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=200, maxLineGap=10)
        
        hsv_values = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 선의 길이 계산
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # 길이가 200 이상인 선만 처리
                if line_length >= 200:
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
                    # 수평 및 수직 선 필터링 (±5도)
                    if (abs(angle) <= 5 or abs(angle) >= 175) or (abs(angle - 90) <= 5 or abs(angle - 270) <= 5):
                        # 선의 색상 추출
                        color = image[y1, x1]
                        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
                        hsv_values.append(hsv_color)
                        print(f"Line: ({x1}, {y1}) to ({x2}, {y2}), Angle: {angle:.2f}, Length: {line_length}, HSV: {hsv_color}")

        # HSV 값들의 평균 계산
        if hsv_values:
            average_hsv = np.mean(hsv_values, axis=0).astype(int)
            print(f"Average HSV: {average_hsv}")    
        
        
    def detect_edges_in_hsv_range(self, img_color, lower_hsv, upper_hsv, \
                                  canny_threshold1=50, canny_threshold2=150, \
                                  hough_threshold=110, min_line_length=80, max_line_gap=150):
        # OpenCV 이미지를 HSV 색공간으로 변환
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        # 주어진 HSV 범위 내의 마스크 생성
        img_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
        
        # img_mask 이미지를 QImage로 변환
        maskQImg = QImage(img_mask.data, img_mask.shape[1], img_mask.shape[0], img_mask.shape[1], QImage.Format_Grayscale8)

        # QImage를 NumPy 배열로 변환
        ptr = maskQImg.bits()
        ptr.setsize(maskQImg.byteCount())
        bytes_per_line = maskQImg.bytesPerLine()
        # QImage가 8비트 그레이스케일 이미지라면 한 픽셀당 바이트 수는 1이 됩니다.
        mask_array = np.frombuffer(ptr, dtype=np.uint8).reshape((maskQImg.height(), bytes_per_line))

        # Canny 엣지 검출
        edges = cv2.Canny(mask_array, canny_threshold1, canny_threshold2)
        
        # Hough 변환을 사용한 선 검출
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, hough_threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        # 90도로 꺾이는 선의 위치 찾기
        if lines  is None:
            print('==== 검출된 line이 없습니다. =============')
            return edges

        sel_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # 각도가 90도 내외인 경우에만 출력 (각도 허용 오차 고려)
            if abs(theta) < 92 and abs(theta) > 88:
                cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 4)
                sel_lines.append((x1,y1,x2,y2))
                # print(f'{i} 2theta={theta} x1, y1, x2, y2 = {x1} {y1} {x2} {y2}')
            # 각도가 수평선
            if abs(theta) < 2 or abs(theta-180) < 2 or abs(theta-90) < 2:
            # if abs(theta) < 2 or abs(theta-180) < 2 or abs(theta-90) < 2:
                cv2.line(edges, (x1, y1), (x2, y2), (255, 0, 0), 4)
                sel_lines.append((x1,y1,x2,y2))
                # print(f'{i} 2theta={theta} x1, y1, x2, y2 = {x1} {y1} {x2} {y2}')

        self.selLines =sel_lines #class 변수로

        return edges

    def draw_points_with_text(self, edges, points):
        cp_edges = edges.copy() # 원본 변경을 피하기 위해
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.5
        font_color = (255, 0, 0)  
        font_thickness = 2
        circle_radius = 20
        circle_color = (255, 0, 0)  
        circle_thickness = -1  # 원 안을 채움

        for point in points:
            x, y = point
            # 이미지에 원 그리기
            cv2.circle(cp_edges, (int(x), int(y)), circle_radius, circle_color, circle_thickness)
            # 좌표 텍스트로 표시
            text = f"({int(x)}, {int(y)})"
            # 텍스트 위치 조정: 원의 중심에서 약간 위로 올립니다.
            text_position = (int(x) - 40, int(y) - 25)
            cv2.putText(cp_edges, text, text_position, font, font_scale, font_color, font_thickness)
        return cp_edges
    
    def is_within_range(self, x, y, tolerance=5):
        for px, py in self.interSections:
            if (abs(px - x) <= tolerance) and (abs(py - y) <= tolerance):
                return 3  # x와 y 모두 범위 내
            elif abs(px - x) <= tolerance:
                return 1  # x만 범위 내
            elif abs(py - y) <= tolerance:
                return 2  # y만 범위 내
        return 0  # 범위 밖

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        
        self.analyze_lines_and_colors(filePath)

        # line_color = self.detect_lines_and_colors(filePath)

        # if line_color is not None:
        #     print(f"Detected line color (HSV): {line_color}")
        # else:
        #     print("No horizontal or vertical lines detected.")

        # average_hsv = self.select_roi(filePath)
        # lower_hsv, upper_hsv = self.define_hsv_range(average_hsv)

        # print("Average HSV:", average_hsv)
        # print("Lower HSV Threshold:", lower_hsv)
        # print("Upper HSV Threshold:", upper_hsv)

        if filePath:
            img_color = cv2.imread(filePath)
            if img_color is not None:
                # OpenCV 이미지를 QImage로 변환
                img_color_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB) 
                bytesPerLine = 3 * img_color_rgb.shape[1]
                qImg = QImage(img_color_rgb.data, img_color_rgb.shape[1], img_color_rgb.shape[0], bytesPerLine, QImage.Format_RGB888)
                self.qImage = qImg        #class 변수 저장

                canny_threshold1=50
                canny_threshold2=150
                hough_threshold=110 
                min_line_length=80 
                max_line_gap=150

                if "aaa.jpg" in filePath:
                    # 회색색의 HSV 범위 정의 ccc.jpg
                    lower_hsv = (140, 100, 100)
                    upper_hsv = (160, 255, 255)
                elif "bbb.jpg" in filePath:
                    # 핑크색의 HSV 범위 정의 aaa.jpg
                    lower_hsv = (6, 109,210)   # Hue , Saturation , Value
                    upper_hsv = (26, 189, 255)  
                else:
                    # 연녹색 HSV 범위 정의 bbb.jpg
                    lower_hsv = (79, 215, 18)
                    upper_hsv = (99, 255, 118)
                

                # dominant_hsv = self.find_dominant_color(img_color_rgb)
                # lower_hsv, upper_hsv = self.get_hsv_range(dominant_hsv)
                # print(f"Image: {filePath} - Dominant HSV: {dominant_hsv}, Lower HSV: {lower_hsv}, Upper HSV: {upper_hsv}")

                msk_edges = self.detect_edges_in_hsv_range(img_color, lower_hsv, upper_hsv, 
                                                       canny_threshold1, canny_threshold2, 
                                                       hough_threshold, min_line_length, max_line_gap)
                if self.selLines == []:
                    return
                
                self.mskEdges = msk_edges      # class 변수저장 원본 저장

                intersections = self.get_intersections(self.selLines)
                self.interSections = intersections        #class 변수 저장
                # 교차점 리스트에서 중복 또는 비슷한 위치 제거
                unique_intersections = self.remove_near_duplicates(intersections , 12)
                # 이미지에 교차점을 그림
                averaged_points = self.average_within_tolerance( unique_intersections, 12)
                # print(averaged_points)

                text_edges = self.draw_points_with_text(self.mskEdges, averaged_points)

                non_overlap_box  = self.find_non_overlapping_boxes(averaged_points)
                self.faceBoxes = non_overlap_box  # class 변수에 저장
                # print(f'non overlapped box count={len(non_overlap_box)}')
                # for box_coord in non_overlap_box:
                #     print(box_coord)

                pixImg = self.draw_boxes_on_image( qImg,non_overlap_box)
                # self.save_boxes(image_folder,qImg,non_overlap_box)

                self.displayImage(pixImg, text_edges)
            else:
                print("이미지를 불러올 수 없습니다.")

    def update_image_display(self):
        if self.mskEdges is None:
            print("이미지 또는 msk_edges가 로드되지 않았습니다.")
            return

        unique_intersections = self.remove_near_duplicates(self.interSections , 12)
                # 이미지에 교차점을 그림
        averaged_points = self.average_within_tolerance( unique_intersections, 12)
                # print(averaged_points)

        text_edges = self.draw_points_with_text(self.mskEdges, averaged_points)
        non_overlap_box = self.find_non_overlapping_boxes(averaged_points)
        self.faceBoxes = non_overlap_box  # class 변수에 저장
        print(f'non overlapped box count={len(non_overlap_box)}')

        pixImg = self.draw_boxes_on_image(self.qImage, non_overlap_box)
        
        self.update_displayImage(pixImg, text_edges)

    def update_displayImage(self,pixImg, text_edges):

        # NumPy 배열의 너비와 높이를 얻습니다.
        edgesHeight, edgesWidth = text_edges.shape
        bytesPerLine = edgesWidth
        edgesQImg = QImage(text_edges.data, edgesWidth, edgesHeight, bytesPerLine, QImage.Format_Grayscale8)
        # QImage로 변환된 edgesQImg를 QPixmap으로 변환
        edgesPixmap = QPixmap.fromImage(edgesQImg)

        # # QLabel에 새로운 QPixmap 설정
        self.imgLabel.setPixmap(pixImg.scaled(self.imgLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.maskImgLabel.setPixmap(edgesPixmap.scaled(self.maskImgLabel.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # 저장된 원본도 바꾼다
        self.imgLabel.setOriginalPixmap(pixImg)
        self.maskImgLabel.setOriginalPixmap(edgesPixmap)
        # 화면 갱신
        self.imgLabel.update()
        self.maskImgLabel.update()

    def toggle_point_in_list(self, x, y):
        circle_radius = 20
        point_found = None

        # 리스트에 존재하는 각 좌표와의 거리를 계산하여 circle_radius 범위 내에 있는지 확인
        for existing_point in self.interSections:
            distance = np.sqrt((existing_point[0] - x) ** 2 + (existing_point[1] - y) ** 2)
            if distance <= circle_radius:
                point_found = existing_point
                break

        if point_found:
            # 범위 내에 좌표가 존재하면 해당 좌표 제거
            self.interSections.remove(point_found)
            print(f"Point {point_found} removed from the list.")
        else:
            # 범위 내에 좌표가 존재하지 않으면 좌표 추가
            self.interSections.append((x, y))
            print(f"Point ({x}, {y}) added to the list.")

    def show_click_count(self,X, Y):
        self.click_count %= 6
        faceName = [name for name, index in faceKeywords.items() if index == self.click_count ][0]
        faceIndex = self.click_count
        self.click_count += 1
         # 팝업 예시 사용
        popup = PopupLabel(self)
        popup_width, popup_height = popup.sizeHint().width(), popup.sizeHint().height()

        # 클릭된 위치의 바로 위 중앙에 팝업을 표시하기 위한 위치 계산
        globalPos = QCursor.pos()
        popup_x = globalPos.x() - popup_width // 2
        popup_y = globalPos.y() - popup_height

        # 전역 좌표를 위젯의 좌표계로 변환
        localPos = self.mapFromGlobal(QPoint(popup_x, popup_y))
        # 팝업 위치 설정
        popup.showWithTimeout(faceName, localPos)
        popup.move(popup_x, popup_y)

        find_box = self.find_box_containing_point(self.faceBoxes, X, Y)
        if find_box is not None:
            saved_image_path = self.save_face_box_as_image(find_box,faceName)
            # self.glWidget.updateTexture(saved_image_path, faceIndex) 
            self.loadBoxFace()
        else:
            print('Not found')

    def find_box_containing_point(self, boxes, globalX, globalY):
        # 박스 리스트에서 각 박스를 순회하며 좌표 검사
        for box in boxes:
            topLeft, bottomRight = box
            topLeftX, topLeftY = topLeft
            bottomRightX, bottomRightY = bottomRight            
            # 전역 좌표를 위젯의 좌표계로 변환
            localPos = self.mapFromGlobal(QPoint(globalX, globalY))
            x, y = localPos.x(), localPos.y()
            # print(topLeftX,globalX,bottomRightX,topLeftY,globalY ,bottomRightY)

            # 좌표가 박스 내에 있는지 확인
            if topLeftX <= globalX <= bottomRightX and topLeftY <= globalY <= bottomRightY:
                return box  # 해당 박스 반환
        return None  # 좌표가 어떤 박스에도 속하지 않는 경우
            
    
    def save_face_box_as_image(self, box, faceName):
        # 박스 좌표 추출
        topLeft, bottomRight = box
        topLeftX, topLeftY = topLeft
        bottomRightX, bottomRightY = bottomRight
        
        # 박스의 너비와 높이 계산
        width = bottomRightX - topLeftX
        height = bottomRightY - topLeftY
        
        # QImage에서 박스 영역을 추출
        cropped_qimage = self.qImage.copy(topLeftX, topLeftY, width, height)
        
        # 저장할 경로 및 파일 이름 설정
        save_path = f'{image_folder}{faceName}.jpg'
        
        # 이미지 저장
        cropped_qimage.save(save_path, 'JPG')

        return save_path  # 저장된 이미지의 경로를 반환합니다.

    def displayImage(self, pixImg, edges):
        screenWidth = QApplication.desktop().screenGeometry().width()
        screenHeight = QApplication.desktop().screenGeometry().height()

        # 이미지 크기 조정을 위한 최대 크기 설정
        maxDisplayWidth = screenWidth * 1.0  # 화면 너비의 80%
        maxDisplayHeight = screenHeight * 1.0  # 화면 높이의 80%
        
        # 원본 이미지와 마스크 이미지의 조정된 크기 계산
        qImgWidth = pixImg.width()
        qImgHeight = pixImg.height()

        # NumPy 배열의 너비와 높이를 얻습니다.
        edgesHeight, edgesWidth = edges.shape
        bytesPerLine = edgesWidth
        edgesQImg = QImage(edges.data, edgesWidth, edgesHeight, bytesPerLine, QImage.Format_Grayscale8)
        # QImage로 변환된 edgesQImg를 QPixmap으로 변환
        edgesPixmap = QPixmap.fromImage(edgesQImg)

        # 최대 표시 크기에 맞춰 이미지 크기 조정
        ratio = min(maxDisplayWidth / (qImgWidth + edgesWidth), maxDisplayHeight / max(qImgHeight, edgesHeight))
        newQImgWidth = int(qImgWidth * ratio)
        newQImgHeight = int(qImgHeight * ratio)
        newMaskQImgWidth = int(edgesWidth * ratio)
        newMaskQImgHeight = int(edgesHeight * ratio)

        # imageWindow가 이미 존재하는지 확인
        if self.imageWindow is None:
            self.imageWindow = QWidget()
            self.imageWindow.setWindowTitle('Image and Mask Preview')
            # self.imageWindow.setFixedSize(newQImgWidth + newMaskQImgWidth, max(newQImgHeight, newMaskQImgHeight))      # imageWindow의 크기를 고정합니다.

            self.layout = QHBoxLayout(self.imageWindow)
        else:
            self.layout.removeWidget(self.imgLabel)  # 기존 위젯 제거
            self.layout.removeWidget(self.maskImgLabel)
       
        # 원본 이미지의 너비와 높이를 저장
        self.imgLabel = ClickableLabel(self, qImgWidth, qImgHeight)
        # 클릭 시그널 연결
        self.imgLabel.clicked.connect(self.handle_click)          

        self.imgLabel.setOriginalPixmap(pixImg)
        self.layout.addWidget(self.imgLabel)
        
        self.maskImgLabel = ClickableLabel(self, edgesWidth, edgesHeight)
        self.maskImgLabel.clicked.connect(self.handle_click)          

        self.maskImgLabel.setOriginalPixmap(edgesPixmap)
        self.layout.addWidget(self.maskImgLabel)
        
        self.imageWindow.setLayout(self.layout)
        self.imageWindow.setGeometry(100, 100, newQImgWidth + newMaskQImgWidth, max(newQImgHeight, newMaskQImgHeight))
        self.imageWindow.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
