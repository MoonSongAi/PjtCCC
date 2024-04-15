
import os

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
from PyQt5.QtCore import QPoint, Qt 
from PyQt5.QtOpenGL import QGLWidget
from config import faceKeywords, image_folder
class GLWidget(QGLWidget):
    def __init__(self, mainWindow):
        super(GLWidget, self).__init__(mainWindow)
        self.mainWindow = mainWindow
        self.textureIDs = [0] * 6  # 텍스처 ID를 저장할 리스트
        self.zoomLevel = -5
        self.cubeWidth, self.cubeHeight, self.cubeDepth = 1.0, 1.0, 1.0
         # 초기 회전값 설정: x, y, z축을 중심으로 회전
        # self.xRot =   self.yRot =  self.zRot = 0  # z축 회전은 0으로 설정

        self.xRot = 30  # x축을 중심으로 30도 회전
        self.yRot = -245  # y축을 중심으로 -45도 회전
        self.zRot = 0  # z축 회전은 0으로 설정
        self.lastPos = QPoint()

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        self.setupTextures()

    def setupTextures(self):
        for faceName ,index in faceKeywords.items():
            filename = os.path.join(image_folder, f'{faceName}.jpg')
            self.loadTextureForFace(filename, index)

        width, depth = self.mainWindow.calculate_dimension_ratios(image_folder)
        self.mainWindow.widthInput.setText("{:.2f}".format(width))
        self.mainWindow.depthInput.setText("{:.2f}".format(depth))


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
    ### add by Yoon 2024 / 04/12 
    def updateTexture(self, imagePath, faceIndex):
        # 기존 텍스처가 있다면 삭제
        if self.textureIDs[faceIndex] != 0:
            glDeleteTextures([self.textureIDs[faceIndex]])
        # 새 텍스처 생성 
        textureID = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, textureID)
        image = Image.open(imagePath)
        
        if faceIndex == 4 or faceIndex == 5:
            image = image.rotate(180, expand=True)
        else:
            image = image.rotate(-90, expand=True)

        image = image.transpose(Image.FLIP_TOP_BOTTOM)  # 이미지를 상하 반전
        ix, iy, image = image.size[0], image.size[1], image.tobytes("raw", "RGBA", 0, -1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        self.textureIDs[faceIndex] = textureID
        self.update()

