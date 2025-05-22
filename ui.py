

import sys
import cv2
import time
import numpy as np
from PIL import Image, ImageQt
from model.mlp import MLP, Dense

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QSizePolicy
from PyQt5.QtGui import QPainter, QImage, QPen
from PyQt5.QtCore import Qt, QSize, QPoint
from PyQt5.QtGui import QColor, QPixmap

# 定义网络结构
layers = [
    Dense(input_size=784, output_size=256, activation='tanh'),
    Dense(input_size=256, output_size=128, activation='relu'),
    Dense(input_size=128, output_size=256, activation='relu'),
    Dense(input_size=256, output_size=10, activation='sigmoid')
]
# 创建新的mlp
mlp = MLP(layers)
# 测试模型
mlp.load_model('result/tl/mpl_2000.npz') # 加载模型权重



class DrawingBoard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_pos = None
        self.drawing = False
        # 创建图像高宽为正方形的黑色图像
        self.image_size = 300
        self.image = QImage(self.image_size, self.image_size, QImage.Format_Grayscale8)
        self.image.fill(Qt.black)

    def sizeHint(self):
        return QSize(self.image_size, self.image_size)

    def minimumSizeHint(self):
        return QSize(self.image_size, self.image_size)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() and Qt.LeftButton and self.drawing:
            painter = QPainter(self.image)
            pen = QPen()
            pen.setWidth(15)
            pen.setColor(Qt.white)
            painter.setPen(pen)
            painter.drawLine(self.last_pos, event.pos())
            self.last_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clear_image(self):
        self.image.fill(Qt.black)
        self.update()
    
    def get_image_data(self):
        # 将QImage转换为numpy数组并调整大小为28x28像素
        img = self.image
        width = img.width()
        height = img.height()
        ptr = img.bits()
        ptr.setsize(height * width)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width))
        arr = cv2.resize(arr, (28, 28), interpolation=cv2.INTER_AREA)
        return arr
        


class PaintBoard(QWidget):
    def __init__(self, Parent = None, Size = QSize(280, 280), Fill = QColor(255,255,255,255)):
        super().__init__(Parent)
    
        # 初始化参数
        self.__size = Size                  # 画板尺寸
        self.__fill = Fill                  # 画板默认填充颜色

        self.__thickness = 15               # 默认画笔粗细
        self.__penColor = Qt.white     # 默认画笔颜色

        self.__begin_point = QPoint()
        self.__end_point = QPoint()

        # 初始化画板界面
        self.__board = QPixmap(self.__size)
        self.__board.fill(Fill) 
        self.setFixedSize(self.__size)
        self.__painter = QPainter()         # 新建绘图工具


    # 清空画板
    def Clear(self):
        self.__board.fill(self.__fill)
        self.update()

    def setBoardFill(self, fill):
        self.__fill = fill
        self.__board.fill(fill)
        self.update()
    
    # 设置画笔颜色    
    def setPenColor(self, color):
        self.__penColor = color

    # 设置画笔粗细    
    def setPenThickness(self, thickness=10):    
        self.__thickness = thickness  

    # 获取画板QImage类型图片
    def getContentAsQImage(self):
        image = self.__board.toImage()
        return image 

    # 双缓冲绘图，绘图事件
    def paintEvent(self, paintEvent):         
        self.__painter.begin(self)
        self.__painter.drawPixmap(0,0,self.__board)
        self.__painter.end()

    def mousePressEvent(self, mouseEvent):
        if mouseEvent.button() == Qt.LeftButton:
            self.__begin_point = mouseEvent.pos()
            self.__end_point = self.__begin_point
            # self.update()

    def mouseMoveEvent(self, mouseEvent):
        if mouseEvent.buttons() == Qt.LeftButton:
            self.__end_point = mouseEvent.pos()

            # 画入缓冲区
            self.__painter.begin(self.__board)
            self.__painter.setPen(QPen(self.__penColor,self.__thickness))  
            self.__painter.drawLine(self.__begin_point, self.__end_point)
            self.__painter.end()

            self.__begin_point = self.__end_point
            self.update()



      
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("手写数字识别")
        self.setGeometry(100, 100, 400, 400)
        # 禁止拉伸窗口大小
        self.setFixedSize(self.size())
        # 禁止窗口最大化
        self.setWindowState(Qt.WindowNoState)
        
        # self.drawing_board = DrawingBoard(self)
        self.paintBoard = PaintBoard(self, Size = QSize(280, 280), Fill = Qt.black)
        
        self.recognition_label = QLabel("预测结果：", self)
        self.clear_button = QPushButton("清除", self)
        self.predict_button = QPushButton("预测", self)

        layout = QVBoxLayout()
        
        # layout.addWidget(self.drawing_board)
        layout.addWidget(self.paintBoard)
        
        layout.addWidget(self.recognition_label)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.predict_button)

        container = QWidget()
        
        container.setLayout(layout)
        
        self.setCentralWidget(container)

        # Set the central widget's size policy to ensure the drawing board is square
        # container.setSizePolicy(
        #     QSizePolicy.Preferred,
        #     QSizePolicy.Preferred
        # )

        # Adjust the main window size to fit the drawing board
        # self.adjustSize()
        
        # self.clear_button.clicked.connect(self.drawing_board.clear_image)
        
        self.clear_button.clicked.connect(self.paintBoard.Clear)
        self.predict_button.clicked.connect(self.predict_digit)

    def predict_digit(self):
        # 获取当前画板内容
        # image_data = self.drawing_board.get_image_data()
        # image_data = image_data / 255.0  # 归一化
        # 使用paintBoard获取当前画板内容
        __img = self.paintBoard.getContentAsQImage()
        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(__img)
        pil_img = pil_img.resize((28, 28), Image.LANCZOS)
        image_data = np.array(pil_img.convert('L')).reshape(28, 28) / 255.0
        # 获取当前时间戳作为文件名事件格式化YYYYMMDDHHMMSS
        # times = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        # cv2.imwrite(f'test/test_{times}.png', image_data * 255)
        # # 这里调用你的MLP模型进行预测
        prediction = mlp.forward(image_data.flatten().reshape(1, -1))
        # print(prediction)
        prediction = prediction.argmax(axis=1)  # 假设mlp是已训练好的模型
        self.recognition_label.setText(f"预测结果：{prediction[0]}")
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())