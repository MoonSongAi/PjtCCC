
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
from PyQt5.QtCore import QPoint, Qt ,pyqtSignal, QTimer 
from PyQt5.QtWidgets import (
    QLabel,
    QToolTip
)
from PyQt5.QtGui import (
    QPainter, 
    QPen, 
    QColor,
)
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
    #여기 clicked 
    clicked = pyqtSignal(int , int , str)  # 사용자 정의 시그널, 클릭된 좌표와 클릭된 버튼("left" 또는 "right")을 전달합니다. x,y값 오른쪽왼쪽클릭
    def __init__(self, original_width, original_height, *args, **kwargs):
        super(ClickableLabel, self).__init__(*args, **kwargs)
        self.original_width = original_width
        self.original_height = original_height
        self.setMouseTracking(True)  # 마우스 추적 활성화 ///////////반드시////////// 마우스 트래킹 CPU가 계속 돌기 떄문에
        self.crosshairPosition = None  # 십자선 위치 초기화 x,y값 선언하기 위해서
        #mousePressEvent 리저브도어 : 정해진 말 (노란색)
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

        # 사용자 정의 시그널을 발생시켜 MainWindow에 클릭 좌표 전달 ///////  x,y값 오른쪽왼쪽클릭
        self.clicked.emit(int(original_x)+2, int(original_y)+2 , button_clicked) #emit : click 이벤트가 발생하면 시그넣읗 던져라 여기 clicked 연동되어야 함

    def mouseMoveEvent(self, event):
        # QLabel의 현재 크기에 맞는 스케일 비율을 계산
        scale_width = self.width() / self.original_width
        scale_height = self.height() / self.original_height

        # 클릭 좌표를 원본 QPixmap 상의 좌표로 변환
        original_x = event.pos().x() / scale_width
        original_y = event.pos().y() / scale_height

        # 십자선 위치 업데이트
        self.crosshairPosition = QPoint(int(original_x), int(original_y))
        self.update()  # 화면 갱신 요청

        tooltipText = f"Mouse at: ({int(original_x)}, {int(original_y)})"
        QToolTip.showText(event.globalPos(), tooltipText, self)

    def paintEvent(self, event):
        super(ClickableLabel, self).paintEvent(event)
        if self.crosshairPosition:
            painter = QPainter(self)
            pen = QPen(QColor(255, 0, 0), 2, Qt.DotLine)  # 검은색, 점선 스타일
            painter.setPen(pen)
            # 스케일 비율 계산
            scale_width = self.width() / self.original_width
            scale_height = self.height() / self.original_height
             # 스케일링된 십자선 위치 계산
            x = int(self.crosshairPosition.x() * scale_width)
            y = int(self.crosshairPosition.y() * scale_height)
            # 십자선 그리기
            painter.drawLine(0, y, self.width(), y)  # 가로선
            painter.drawLine(x, 0, x, self.height())  # 세로선

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
    
