# pip install PyOpenGL PyOpenGL_accelerate
# pip install PyQt5

import os
import sys
import cv2
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
from PyQt5.QtCore import QPoint, Qt ,pyqtSignal, QTimer 
from PyQt5.QtOpenGL import QGLWidget
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
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor,QCursor

# 모듈 최상단에 전역 변수 정의
faceKeywords = {
    "앞면": 0,
    "우측면": 1,
    "뒷면": 2,
    "좌측면": 3,
    "상단면": 4,
    "하단면": 5,
}
class PopupLabel(QLabel):
    def __init__(self, parent=None):
        super(PopupLabel, self).__init__(parent)
        self.setStyleSheet("background-color: white; border: 1px solid black; padding: 2px;")
        self.adjustSize()
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def showWithTimeout(self, text, pos, timeout=1000):
        self.setText(text)
        self.adjustSize()
        # 전역 좌표를 위젯의 좌표계로 변환
        localPos = self.parent().mapFromGlobal(pos)
        # 팝업 위치 조정
        self.move(localPos + QPoint(10, -10))  # 예시로 10, -10 만큼 오프셋 추가
        self.show()
        QTimer.singleShot(timeout, self.close)

class ClickableLabel(QLabel): 
    clicked = pyqtSignal(int , int , str)  # 사용자 정의 시그널, 클릭된 좌표와 클릭된 버튼("left" 또는 "right")을 전달합니다.

    def __init__(self, original_width, original_height, *args, **kwargs):
        super(ClickableLabel, self).__init__(*args, **kwargs)
        self.original_width = original_width
        self.original_height = original_height

    def mousePressEvent(self, event):
        # QLabel의 현재 크기에 맞는 스케일 비율을 계산
        scale_width = self.width() / self.original_width
        scale_height = self.height() / self.original_height

        # 클릭 좌표를 원본 QPixmap 상의 좌표로 변환
        original_x = event.pos().x() / scale_width
        original_y = event.pos().y() / scale_height

        # 클릭된 마우스 버튼 확인
        button_clicked = ""
        if event.button() == Qt.LeftButton:
            button_clicked = "left"
        elif event.button() == Qt.RightButton:
            button_clicked = "right"

        # 사용자 정의 시그널을 발생시켜 MainWindow에 클릭 좌표 전달
        self.clicked.emit(int(original_x)+2, int(original_y)+2 , button_clicked)

class BoxCalculator:
    def __init__(self, boxes):
        self.boxes = boxes
    
    def calculate_dimensions(self):
        # 가장 넓은 박스 찾기
        front_box, max_area = self.find_largest_box()

        # "앞면" 박스의 가로와 세로 길이를 계산
        front_width = front_box[1][0] - front_box[0][0]
        front_height = front_box[1][1] - front_box[0][1]

        # 앞면 중 가장 긴 쪽을 높이로 정하고 나머지를 가로로 정함
        height, width = max(front_width, front_height), min(front_width, front_height)

        # 깊이 계산
        depth = self.calculate_depth(front_width)

        return height, width, depth
    
    def find_largest_box(self):
        max_area = 0
        front_box = None
        for box in self.boxes:
            width = box[1][0] - box[0][0]
            height = box[1][1] - box[0][1]
            area = width * height
            if area > max_area:
                max_area = area
                front_box = box
        return front_box, max_area
    
    def calculate_depth(self, front_width):
        depths = []
        for box in self.boxes:
            box_width = box[1][0] - box[0][0]
            depth = abs(front_width - box_width)
            depths.append(depth)
        return sum(depths) / len(depths) if depths else 0
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("test2")
        self.setGeometry(100, 100, 800, 600)
        self.imagePaths = [None] * 6
        self.initUI()
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
        btnBatchSelect.clicked.connect(self.openMultipleImagesFileDialog)
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

    def loadImage(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
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

                # 핑크색의 HSV 범위 정의 aaa.jpg
                # lower_hsv = (140, 100, 100)
                # upper_hsv = (160, 255, 255)
                # 연녹색 HSV 범위 정의 bbb.jpg
                # lower_hsv = (6, 109,210)   # Hue , Saturation , Value
                # upper_hsv = (26, 189, 255)  
                # 회색색의 HSV 범위 정의 ccc.jpg
                lower_hsv = (79, 215, 18)
                upper_hsv = (99, 255, 118)
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
                print(f'non overlapped box count={len(non_overlap_box)}')
                for box_coord in non_overlap_box:
                    print(box_coord)

                # calculator = BoxCalculator(non_overlap_box)
                # height, width, depth = calculator.calculate_dimensions()
                # print(f"Height: {height}, Width: {width}, Depth: {depth}")

                pixImg = self.draw_boxes_on_image( qImg,non_overlap_box)
                # self.save_boxes('./box_images',qImg,non_overlap_box)

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
        screenWidth = QApplication.desktop().screenGeometry().width()
        screenHeight = QApplication.desktop().screenGeometry().height()

        # 이미지 크기 조정을 위한 최대 크기 설정
        maxDisplayWidth = screenWidth * 1  # 화면 너비의 100%
        maxDisplayHeight = screenHeight * 1  # 화면 높이의 100%
        
        # 원본 이미지와 마스크 이미지의 조정된 크기 계산
        qImgWidth = pixImg.width()
        qImgHeight = pixImg.height()

        # NumPy 배열의 너비와 높이를 얻습니다.
        edgesHeight, edgesWidth = text_edges.shape
        bytesPerLine = edgesWidth
        edgesQImg = QImage(text_edges.data, edgesWidth, edgesHeight, bytesPerLine, QImage.Format_Grayscale8)
        # QImage로 변환된 edgesQImg를 QPixmap으로 변환
        edgesPixmap = QPixmap.fromImage(edgesQImg)

        # 최대 표시 크기에 맞춰 이미지 크기 조정
        ratio = min(maxDisplayWidth / (qImgWidth + edgesWidth), maxDisplayHeight / max(qImgHeight, edgesHeight))
        newQImgWidth = int(qImgWidth * ratio)
        newQImgHeight = int(qImgHeight * ratio)
        newMaskQImgWidth = int(edgesWidth * ratio)
        newMaskQImgHeight = int(edgesHeight * ratio)
        

        # QLabel에 새로운 QPixmap 설정
        self.imgLabel.setPixmap(pixImg.scaled(newQImgWidth, newQImgHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.maskImgLabel.setPixmap(edgesPixmap.scaled(newMaskQImgWidth, newMaskQImgHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation))

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
    def show_click_count(self,globalX, globalY):
        self.click_count %= 6
        faceName = [name for name, index in faceKeywords.items() if index == self.click_count ][0]
        self.click_count += 1
         # 팝업 예시 사용
        popup = PopupLabel(self)
        popup_width, popup_height = popup.sizeHint().width(), popup.sizeHint().height()

        # 클릭된 위치의 바로 위 중앙에 팝업을 표시하기 위한 위치 계산
        popup_x = globalX - popup_width // 2
        popup_y = globalY - popup_height

        # 전역 좌표를 위젯의 좌표계로 변환
        localPos = self.mapFromGlobal(QPoint(popup_x, popup_y))
        # 팝업 위치 설정
        popup.showWithTimeout(faceName, localPos)
        popup.move(popup_x, popup_y)

        find_box = self.find_box_containing_point(self.faceBoxes, localPos.x(), localPos.y())
        if find_box is not None:
            print(find_box)
        else:
            print('Mot found')

    def find_box_containing_point(self, boxes, globalX, globalY):
        # 박스 리스트에서 각 박스를 순회하며 좌표 검사
        for box in boxes:
            topLeft, bottomRight = box
            topLeftX, topLeftY = topLeft
            bottomRightX, bottomRightY = bottomRight

            print(topLeft,bottomRight,globalX, globalY)
            
            # 전역 좌표를 위젯의 좌표계로 변환
            localPos = self.mapFromGlobal(QPoint(globalX, globalY))
            x, y = localPos.x(), localPos.y()
            
            # 좌표가 박스 내에 있는지 확인
            if topLeftX <= x <= bottomRightX and topLeftY <= y <= bottomRightY:
                return box  # 해당 박스 반환
        return None  # 좌표가 어떤 박스에도 속하지 않는 경우
    
    def handle_click(self, x, y,button_clicked):
       # 클릭된 좌표와 마우스 버튼 정보 처리
        if button_clicked == "left":
            self.toggle_point_in_list(x, y)
            self.update_image_display()
        elif button_clicked == "right":
            print(f"Right button clicked at: ({x}, {y})")
            globalPos = QCursor.pos()
            self.show_click_count(globalPos.x(), globalPos.y())
        else:
            print(f"Clicked at: ({x}, {y}) without recognized button")

    def displayImage(self, pixImg, edges):
        screenWidth = QApplication.desktop().screenGeometry().width()
        screenHeight = QApplication.desktop().screenGeometry().height()

        # 이미지 크기 조정을 위한 최대 크기 설정
        maxDisplayWidth = screenWidth * 1  # 화면 너비의 80%
        maxDisplayHeight = screenHeight * 1  # 화면 높이의 80%
        
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
            self.layout = QHBoxLayout(self.imageWindow)
        else:
            self.layout.removeWidget(self.imgLabel)  # 기존 위젯 제거
            self.layout.removeWidget(self.maskImgLabel)


        
        # imgLabel = QLabel()
        # edgesPixmap을 표시하는 라벨도 ClickableLabel로 만들어 클릭 가능하게 함
        # 원본 이미지의 너비와 높이를 저장
        self.imgLabel = ClickableLabel(qImgWidth, qImgHeight)
        self.imgLabel.clicked.connect(self.handle_click)  # 클릭 시그널 연결
        self.imgLabel.setPixmap(pixImg.scaled(newQImgWidth, newQImgHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.layout.addWidget(self.imgLabel)
        
        self.maskImgLabel = QLabel()
        # maskImgLabel = ClickableLabel(edgesWidth, edgesHeight)
        # maskImgLabel.clicked.connect(self.handle_click)  # 클릭 시그널 연결
        self.maskImgLabel.setPixmap(edgesPixmap.scaled(newMaskQImgWidth, newMaskQImgHeight, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.layout.addWidget(self.maskImgLabel)
        
        self.imageWindow.setLayout(self.layout)
        self.imageWindow.setGeometry(100, 100, newQImgWidth + newMaskQImgWidth, max(newQImgHeight, newMaskQImgHeight))
        self.imageWindow.show()

class GLWidget(QGLWidget):
    def __init__(self, mainWindow):
        super(GLWidget, self).__init__(mainWindow)
        self.mainWindow = mainWindow
        self.textureIDs = [0] * 6  # 텍스처 ID를 저장할 리스트
        self.zoomLevel = -5
        self.cubeWidth, self.cubeHeight, self.cubeDepth = 1.0, 1.0, 1.0
        self.xRot = self.yRot = self.zRot = 0
        self.lastPos = QPoint()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

    def loadTextureForFace(self, imagePath, faceIndex):
        textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, textureID)
        image = Image.open(imagePath)

        if faceIndex == 4 or faceIndex == 5:
            image = image.rotate(180, expand=True)
        else:
            image = image.rotate(-90, expand=True)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        ix, iy, image = (
            image.size[0],
            image.size[1],
            image.tobytes("raw", "RGBA", 0, -1),
        )
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        self.textureIDs[faceIndex] = textureID
        self.update()

    def wheelEvent(self, event):

        delta = event.angleDelta().y() / 120
        self.zoomLevel += delta
        self.update()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoomLevel)
        glRotatef(self.xRot, 1.0, 0.0, 0.0)
        glRotatef(self.yRot, 0.0, 1.0, 0.0)
        glRotatef(self.zRot, 0.0, 0.0, 1.0)

        self.drawCube()

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, float(w) / h, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def updateCube(self):
        try:

            self.cubeWidth = float(self.mainWindow.widthInput.text())
            self.cubeHeight = float(self.mainWindow.heightInput.text())
            self.cubeDepth = float(self.mainWindow.depthInput.text())
        except ValueError:

            return
        self.update()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.x() - self.lastPos.x()
        dy = event.y() - self.lastPos.y()

        if event.buttons() & Qt.LeftButton:
            self.xRot += dy
            self.yRot += dx
        elif event.buttons() & Qt.RightButton:
            self.xRot += dy
            self.zRot += dx

        self.lastPos = event.pos()
        self.update()

    def drawCube(self):

        vertices = [
            [self.cubeWidth, self.cubeHeight, -self.cubeDepth],
            [self.cubeWidth, -self.cubeHeight, -self.cubeDepth],
            [-self.cubeWidth, -self.cubeHeight, -self.cubeDepth],
            [-self.cubeWidth, self.cubeHeight, -self.cubeDepth],
            [self.cubeWidth, self.cubeHeight, self.cubeDepth],
            [self.cubeWidth, -self.cubeHeight, self.cubeDepth],
            [-self.cubeWidth, -self.cubeHeight, self.cubeDepth],
            [-self.cubeWidth, self.cubeHeight, self.cubeDepth],
        ]

        faces = [
            [0, 1, 2, 3],
            [3, 2, 6, 7],
            [7, 6, 5, 4],
            [4, 5, 1, 0],
            [5, 6, 2, 1],
            [7, 4, 0, 3],
        ]

        # Define colors for each face
        colors = [
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1],  # Blue
            [1, 1, 0],  # Yellow
            [1, 0, 1],  # Magenta
            [0, 1, 1],  # Cyan
        ]
        texCoords = [(1, 0), (0, 0), (0, 1), (1, 1)]

        for i, face in enumerate(faces):
            glBindTexture(GL_TEXTURE_2D, self.textureIDs[i])
            glBegin(GL_QUADS)
            for j, vertex in enumerate(face):
                glTexCoord2f(*texCoords[j % 4])
                glVertex3fv(vertices[vertex])
            glEnd()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
