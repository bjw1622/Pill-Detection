from yolov5 import detect
from east import text_detection, pretreat
from ezocr import madeOcr
from keras_text import keras
from color_detect import color
from deep_text_recognition_benchmark import demo
from extract_csv import csv

import cv2
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

def keras_running(img):
    keras_text = keras.keras_run(img)
    if keras_text == None or '':
        print('detect Nothing')
        keras_text = 'None'
    return keras_text

def ezocr_running(img):
    try:
        ocr = madeOcr.main_use(img)
    except:
        print('detect Nothing')
        ocr = 'None'
    return ocr

def bmocr_running(img):
    bench_mark_text = demo.demo()
    if bench_mark_text == None:
        bench_mark_text = 'None'
    return bench_mark_text

class DCA(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Drug Classification App')
        self.setWindowIcon((QIcon('pill.png')))
        self.setGeometry(100, 100, 1200, 900)
        
        self.csv_file = "D:/yolo/pill.csv"
        self.result = []
        self.textlist = []

        self.layout = QVBoxLayout(self)

        #탭 생성
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.resize(1200, 900)

        #탭 추가
        self.tabs.addTab(self.tab1, "Image and Model")
        self.tabs.addTab(self.tab2, "Run Models")
        self.tabs.addTab(self.tab3, "Result")

        #탭 1
        #이미지 관련 버튼, 라벨 추가
        self.ImageButton = QPushButton('Open Image')
        self.ImageLabel = QLabel()
        self.ImageLabel.setStyleSheet("background-color: #FFFFFF;"
                                      "border-style: solid;"
                                      "border-width: 1px;"
                                      "border-color: #000000")
        self.ImageLabel.setMaximumHeight(20)
        self.ImageButton.clicked.connect(self.ImageButtonClicked)
        self.ImageButton.setMaximumWidth(100)

        #YOLO 모델 불러오기
        self.Yolo_Model_Button = QPushButton('Open Yolo Model')
        self.Yolo_Model_Label = QLabel()
        self.Yolo_Model_Label.setStyleSheet("background-color: #FFFFFF;"
                                     "border-style: solid;"
                                     "border-width: 1px;"
                                     "border-color: #000000")
        self.Yolo_Model_Label.setMaximumHeight(20)
        self.Yolo_Model_Button.clicked.connect(self.YoloButtonClicked)
        self.Yolo_Model_Button.setMaximumWidth(100)

        #이미지 보여주기
        self.LoadButton = QPushButton('Show Image')
        self.ImageMap = QPixmap()
        self.Img = QLabel()
        self.Img.setStyleSheet("background-color: #FFFFFF;"
                               "border-style: solid;"
                               "border-width: 1px;"
                               "border-color: #000000")
        self.LoadButton.clicked.connect(self.LoadImage)
        self.LoadButton.setMaximumWidth(100)

        #이미지 경로 버튼, 라벨 추가
        self.image_layout = QHBoxLayout()
        self.image_layout.addWidget(self.ImageButton)
        self.image_layout.addWidget(self.ImageLabel)

        #YOLO 모델 경로 버튼, 라벨 추가
        self.yolo_layout = QHBoxLayout()
        self.yolo_layout.addWidget(self.Yolo_Model_Button)
        self.yolo_layout.addWidget(self.Yolo_Model_Label)

        #이미지 보여주기 버튼
        self.show_layout = QHBoxLayout()
        self.show_layout.addWidget(self.LoadButton)
        self.show_layout.addWidget(self.Img)

        #메인 레이아웃에 나머지 레이아웃 넣기
        self.tab1.layout = QVBoxLayout()
        self.tab1.layout.addLayout(self.image_layout)
        self.tab1.layout.addLayout(self.yolo_layout)
        self.tab1.layout.addLayout(self.show_layout)

        #탭에 레이아웃 설정하기
        self.tab1.setLayout(self.tab1.layout)

        #탭2
        #빈 라벨, 버튼
        self.EMP_Label = QLabel()
        self.EMP_Label.setMaximumWidth(100)
        self.EMP_Label.setMaximumHeight(20)
        self.EMP_Button = QPushButton()
        self.EMP_Button.setMaximumWidth(100)
        self.EMP_Button.setMaximumHeight(20)
        self.EMP_Button.hide()

        #YOLO 이미지
        #RunYolo
        self.YOLO_EXE_Button = QPushButton('YOLO')
        self.YOLO_IMG_Label = QLabel()
        self.YOLO_IMG = QPixmap()
        self.YOLO_IMG_Label.setStyleSheet("background-color: #FFFFFF;"
                                          "border-style: solid;"
                                          "border-width: 1px;"
                                          "border-color: #000000")
        self.YOLO_EXE_Button.clicked.connect(self.RunYolo)
        self.YOLO_EXE_Button.setMaximumWidth(100)
        self.YOLO_IMG_Label.setMaximumHeight(400)
        self.YOLO_IMG_Label.resize(300, 200)

        #EAST 이미지
        self.EAST_EXE_Button = QPushButton("EAST")
        self.EAST_IMG_Label = QLabel()
        self.EAST_IMG = QPixmap()
        self.EAST_IMG_Label.setStyleSheet("background-color: #FFFFFF;"
                                          "border-style: solid;"
                                          "border-width: 1px;"
                                          "border-color: #000000")
        self.EAST_EXE_Button.clicked.connect(self.RunEast)
        self.EAST_EXE_Button.setMaximumWidth(100)
        self.EAST_IMG_Label.setMaximumHeight(400)
        self.EAST_IMG_Label.resize(300, 200)

        #CV 실행
        self.CV_EXE_Button = QPushButton('CV')
        self.CV_Label = QLabel()
        self.CV_Label.setStyleSheet("background-color: #FFFFFF;"
                                    "border-style: solid;"
                                    "border-width: 1px;"
                                    "border-color: #000000")
        self.CV_EXE_Button.clicked.connect(self.RunCV)
        self.CV_EXE_Button.setMaximumWidth(100)
        self.CV_Label.setMaximumWidth(100)
        self.CV_Label.setMaximumHeight(20)
        
        #YOLO - EZOCR
        self.YOLO_EZOCR_EXE_Button = QPushButton('Y_EZOCR')
        self.YOLO_EZOCR_Label = QLabel()
        self.YOLO_EZOCR_Label.setStyleSheet("background-color: #FFFFFF;"
                                            "border-style: solid;"
                                            "border-width: 1px;"
                                            "border-color: #000000")
        self.YOLO_EZOCR_EXE_Button.clicked.connect(self.RunYEZOCR)
        self.YOLO_EZOCR_EXE_Button.setMaximumWidth(100)
        self.YOLO_EZOCR_Label.setMaximumWidth(100)
        self.YOLO_EZOCR_Label.setMaximumHeight(20)

        #YOLO - KRSOCR
        self.YOLO_KRSOCR_EXE_Button = QPushButton('Y_KRSOCR')
        self.YOLO_KRSOCR_Label = QLabel()
        self.YOLO_KRSOCR_Label.setStyleSheet("background-color: #FFFFFF;"
                                             "border-style: solid;"
                                             "border-width: 1px;"
                                             "border-color: #000000")
        self.YOLO_KRSOCR_EXE_Button.clicked.connect(self.RunYKRSOCR)
        self.YOLO_KRSOCR_EXE_Button.setMaximumWidth(100)
        self.YOLO_KRSOCR_Label.setMaximumWidth(100)
        self.YOLO_KRSOCR_Label.setMaximumHeight(20)

        #YOLO - BMOCR
        self.YOLO_BMOCR_EXE_Button = QPushButton('Y_BMOCR')
        self.YOLO_BMOCR_Label = QLabel()
        self.YOLO_BMOCR_Label.setStyleSheet("background-color: #FFFFFF;"
                                             "border-style: solid;"
                                             "border-width: 1px;"
                                             "border-color: #000000")
        self.YOLO_BMOCR_EXE_Button.clicked.connect(self.RunYBMOCR)
        self.YOLO_BMOCR_EXE_Button.setMaximumWidth(100)
        self.YOLO_BMOCR_Label.setMaximumWidth(100)
        self.YOLO_BMOCR_Label.setMaximumHeight(20)

        #EAST - EZOCR
        self.EAST_EZOCR_EXE_Button = QPushButton('E_EZOCR')
        self.EAST_EZOCR_Label = QLabel()
        self.EAST_EZOCR_Label.setStyleSheet("background-color: #FFFFFF;"
                                            "border-style: solid;"
                                            "border-width: 1px;"
                                            "border-color: #000000")
        self.EAST_EZOCR_EXE_Button.clicked.connect(self.RunEEZOCR)
        self.EAST_EZOCR_EXE_Button.setMaximumWidth(100)
        self.EAST_EZOCR_Label.setMaximumWidth(100)
        self.EAST_EZOCR_Label.setMaximumHeight(20)

        #EAST - KRSOCR
        self.EAST_KRSOCR_EXE_Button = QPushButton('E_KRSOCR')
        self.EAST_KRSOCR_Label = QLabel()
        self.EAST_KRSOCR_Label.setStyleSheet("background-color: #FFFFFF;"
                                             "border-style: solid;"
                                             "border-width: 1px;"
                                             "border-color: #000000")
        self.EAST_KRSOCR_EXE_Button.clicked.connect(self.RunEKRSOCR)
        self.EAST_KRSOCR_EXE_Button.setMaximumWidth(100)
        self.EAST_KRSOCR_Label.setMaximumWidth(100)
        self.EAST_KRSOCR_Label.setMaximumHeight(20)

        #EAST - BMOCR
        self.EAST_BMOCR_EXE_Button = QPushButton("E_BMOCR")
        self.EAST_BMOCR_Label = QLabel()
        self.EAST_BMOCR_Label.setStyleSheet("background-color: #FFFFFF;"
                                            "border-style: solid;"
                                            "border-width: 1px;"
                                            "border-color: #000000")
        self.EAST_BMOCR_EXE_Button.clicked.connect(self.RUNEBMOCR)
        self.EAST_BMOCR_EXE_Button.setMaximumWidth(100)
        self.EAST_BMOCR_Label.setMaximumWidth(100)
        self.EAST_BMOCR_Label.setMaximumHeight(20)

        #YOLO, EAST Layout
        self.Y_E_Layout = QHBoxLayout()
        self.Y_E_Layout.addWidget(self.YOLO_EXE_Button)
        self.Y_E_Layout.addWidget(self.YOLO_IMG_Label)
        self.Y_E_Layout.addWidget(self.EAST_EXE_Button)
        self.Y_E_Layout.addWidget(self.EAST_IMG_Label)

        #CV Layout
        self.CV_Layout = QHBoxLayout()
        self.CV_Layout.addWidget(self.CV_EXE_Button)
        self.CV_Layout.addWidget(self.CV_Label)
        self.CV_Layout.addWidget(self.EMP_Label)
        self.CV_Layout.addWidget(self.EMP_Label)

        #EZOCR Layout
        self.EZOCR_Layout = QHBoxLayout()
        self.EZOCR_Layout.addWidget(self.YOLO_EZOCR_EXE_Button)
        self.EZOCR_Layout.addWidget(self.YOLO_EZOCR_Label)
        self.EZOCR_Layout.addWidget(self.EAST_EZOCR_EXE_Button)
        self.EZOCR_Layout.addWidget(self.EAST_EZOCR_Label)

        #KESOCR Layout
        self.KRSOCR_Layout = QHBoxLayout()
        self.KRSOCR_Layout.addWidget(self.YOLO_KRSOCR_EXE_Button)
        self.KRSOCR_Layout.addWidget(self.YOLO_KRSOCR_Label)
        self.KRSOCR_Layout.addWidget(self.EAST_KRSOCR_EXE_Button)
        self.KRSOCR_Layout.addWidget(self.EAST_KRSOCR_Label)

        #BMOCR Layout
        self.BMOCR_Layout = QHBoxLayout()
        self.BMOCR_Layout.addWidget(self.YOLO_BMOCR_EXE_Button)
        self.BMOCR_Layout.addWidget(self.YOLO_BMOCR_Label)
        self.BMOCR_Layout.addWidget(self.EAST_BMOCR_EXE_Button)
        self.BMOCR_Layout.addWidget(self.EAST_BMOCR_Label)

        #메인 레이아웃에 나머지 레이아웃 넣기
        self.tab2.layout = QVBoxLayout()
        self.tab2.layout.addLayout(self.Y_E_Layout)
        self.tab2.layout.addLayout(self.CV_Layout)
        self.tab2.layout.addLayout(self.EZOCR_Layout)
        self.tab2.layout.addLayout(self.KRSOCR_Layout)
        self.tab2.layout.addLayout(self.BMOCR_Layout)

        #탭에 레이아웃 설정하기
        self.tab2.setLayout(self.tab2.layout)

        #탭3
        #맨위 사진 나오게하기 + 버튼
        self.Result_Button = QPushButton('결과 보기')
        self.Result_image_Label = QLabel()
        self.Result_image = QPixmap()
        self.Result_image_Label.setStyleSheet("background-color: #FFFFFF;"
                                              "border-style: solid;"
                                              "border-width: 1px;"
                                              "border-color: #000000")
        self.Result_Button.clicked.connect(self.set_result)
        self.Result_Button.setMaximumWidth(100)
        self.Result_image_Label.setMaximumWidth(600)
        self.Result_image_Label.resize(600, 400)

        #표 만들기
        self.pill_table1_1 = QLabel()
        self.pill_table1_2 = QLabel()
        self.pill_table1_4 = QLabel()
        self.pill_table1_5 = QLabel()
        self.pill_table1_6 = QLabel()

        self.pill_table1_1.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")
        self.pill_table1_2.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")
        self.pill_table1_4.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")
        self.pill_table1_5.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")                                 
        self.pill_table1_6.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")

        self.pill_table1_1.setText('약품 이름')
        self.pill_table1_2.setText('표시 앞')
        self.pill_table1_4.setText('색상')
        self.pill_table1_5.setText('효능')
        self.pill_table1_6.setText('용량')

        self.pill_table1_1.setMaximumWidth(100)
        self.pill_table1_2.setMaximumWidth(100)
        self.pill_table1_4.setMaximumWidth(100)
        self.pill_table1_5.setMaximumWidth(100)
        self.pill_table1_6.setMaximumWidth(100)

        self.pill_table2_1 = QLabel()
        self.pill_table2_2 = QLabel()
        self.pill_table2_4 = QLabel()
        self.pill_table2_5 = QLabel()
        self.pill_table2_6 = QLabel()

        self.pill_table2_1.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")
        self.pill_table2_2.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")
        self.pill_table2_4.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")
        self.pill_table2_5.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")
        self.pill_table2_6.setStyleSheet("background-color: #FFFFFF;"
                                         "border-style: solid;"
                                         "border-width: 1px;"
                                         "border-color: #000000")

        self.pill_table2_1.setMaximumWidth(300)
        self.pill_table2_2.setMaximumWidth(300)
        self.pill_table2_4.setMaximumWidth(300)
        self.pill_table2_5.setMaximumWidth(300)
        self.pill_table2_6.setMaximumWidth(300)
        #스크롤 바
        # self.Scroll_Bar = QScrollArea()
        # self.Scroll_Bar.setWidget(self.pill_table)

        #레이아웃 만들기
        self.result_layout = QVBoxLayout()
        self.result_layout.addWidget(self.Result_Button)
        self.result_layout.addWidget(self.Result_image_Label)

        self.table_layout_1 = QVBoxLayout()
        self.table_layout_1.addWidget(self.pill_table1_1)
        self.table_layout_1.addWidget(self.pill_table1_2)
        self.table_layout_1.addWidget(self.pill_table1_4)
        self.table_layout_1.addWidget(self.pill_table1_5)
        self.table_layout_1.addWidget(self.pill_table1_6)

        self.table_layout_2 = QVBoxLayout()
        self.table_layout_2.addWidget(self.pill_table2_1)
        self.table_layout_2.addWidget(self.pill_table2_2)
        self.table_layout_2.addWidget(self.pill_table2_4)
        self.table_layout_2.addWidget(self.pill_table2_5)
        self.table_layout_2.addWidget(self.pill_table2_6)

        #메인 레이아웃에 나머지 레이아웃 넣기
        self.tab3.Greedlayout = QHBoxLayout()
        self.tab3.Greedlayout.addLayout(self.result_layout)
        self.tab3.Greedlayout.addLayout(self.table_layout_1)
        self.tab3.Greedlayout.addLayout(self.table_layout_2)

        self.tab3.setLayout(self.tab3.Greedlayout)

        #메인 레이아웃에 탭 추가
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        #show
        self.show()
    

    #Image 불러오기 버튼
    def ImageButtonClicked(self):
        fname = QFileDialog.getOpenFileName(self)
        self.ImageLabel.setText(fname[0])

    #YOLO 모델 불러오기 버튼
    def YoloButtonClicked(self):
        fname = QFileDialog.getOpenFileName(self)
        self.Yolo_Model_Label.setText(fname[0])

    #Image 보여주기 버튼
    def LoadImage(self):
        self.ImageMap.load(self.ImageLabel.text())
        self.ImageMap = self.ImageMap.scaledToWidth(600)
        self.Img.setPixmap(self.ImageMap)

    def RunYolo(self):
        img = cv2.imread(self.ImageLabel.text(), 1)
        yolo_label = detect.yolo_run(weights=self.Yolo_Model_Label.text(), source= self.ImageLabel.text(), save_txt= True)
        if yolo_label != None:
            min_x = int(yolo_label[2])
            max_x = int(yolo_label[4])
            min_y = int(yolo_label[1])
            max_y = int(yolo_label[3])

            self.yolo_roi = img[min_x:max_x, min_y:max_y].copy()
            h,w,c = self.yolo_roi.shape
            qImg = QImage(self.yolo_roi.data, w, h, w*c, QImage.Format_RGB888)
            self.YOLO_IMG = QPixmap.fromImage(qImg)
            self.YOLO_IMG = self.YOLO_IMG.scaledToHeight(200)
            self.YOLO_IMG_Label.setPixmap(self.YOLO_IMG)
            height, width = self.yolo_roi.shape[:2]
            M = cv2.getRotationMatrix2D((width/2, height/2), 0, 0.4)
            self.yolo_roi = cv2.warpAffine(self.yolo_roi, M, (width, height))
            cv2.imwrite('D:/yolo/deep_text_recognition_benchmark/demo_image/east_roi.png', self.yolo_roi)
        else:
            self.YOLO_IMG_Label.setText('No Detection')
        
    def RunEast(self):
        east = "D:/yolo/east/frozen_east_text_detection.pb"
        treated_img = pretreat.treat(self.yolo_roi)
        try:
            east_label = text_detection.east_run(treated_img, east)
            for i in east_label:
                min_x = i[0]
                max_x = i[2]
                min_y = i[1]
                max_y = i[3]

                self.east_roi = self.yolo_roi[min_y:max_y, min_x:max_x].copy()
                cv2.imwrite('D:/yolo/deep_text_recognition_benchmark/demo_image/east_roi.png', self.east_roi)
            h,w,c = self.east_roi.shape
            qImg = QImage(self.east_roi.data, w, h, w*c, QImage.Format_RGB888)
            self.EAST_IMG = QPixmap.fromImage(qImg)
            # self.EAST_IMG = self.EAST_IMG.scaledTowidth(200)
            self.EAST_IMG_Label.setPixmap(self.EAST_IMG)
        except:
            self.EAST_IMG_Label.setText('No Detection')
        

    def RunCV(self):
        self.color_dct = color.color_run(self.yolo_roi.copy())
        self.CV_Label.setText(self.color_dct)

    def RunYEZOCR(self):
        self.yolo_ocr = ezocr_running(self.yolo_roi)
        self.YOLO_EZOCR_Label.setText(self.yolo_ocr)
        self.textlist.append(self.yolo_ocr)

    def RunYKRSOCR(self):
        self.keras_yolo_text = keras_running(self.yolo_roi)
        self.YOLO_KRSOCR_Label.setText(self.keras_yolo_text)
        self.textlist.append(self.keras_yolo_text)

    def RunYBMOCR(self):
        self.bmocr_yolo_text = bmocr_running(self.yolo_roi)
        self.YOLO_BMOCR_Label.setText(self.bmocr_yolo_text)
        self.textlist.append(self.bmocr_yolo_text)

    def RunEEZOCR(self):
        self.east_ocr = ezocr_running(self.east_roi)
        self.EAST_EZOCR_Label.setText(self.east_ocr)
        self.textlist.append(self.east_ocr)

    def RunEKRSOCR(self):
        self.keras_east_text = keras_running(self.east_roi)
        self.EAST_KRSOCR_Label.setText(self.keras_east_text)
        self.textlist.append(self.keras_east_text)

    def RUNEBMOCR(self):
        self.bmocr_east_text = bmocr_running(self.east_roi)
        self.EAST_BMOCR_Label.setText(self.bmocr_east_text)
        self.textlist.append(self.bmocr_east_text)

    def set_result(self):
        for i in self.textlist:
            self.result.append(csv.main(self.csv_file, self.color_dct, i))
        
        image = 'D:/yolo/images/'
        result = self.result[0][0]
        try:
            self.pill_table2_1.setText(result[0])
            self.pill_table2_2.setText(result[2])
            self.pill_table2_4.setText(result[3])
            self.pill_table2_5.setText(result[4])
            self.pill_table2_6.setText(result[5])
            result_image = cv2.imread(image + result[1], 1)
            h,w,c = result_image.shape
            qImg = QImage(result_image.data, w, h, w*c, QImage.Format_RGB888)
            self.Result_image = QPixmap.fromImage(qImg)
            self.Result_image_Label.setPixmap(self.Result_image)
        except:
            self.Result_image_Label.setText("Nothing Found")


# img = "D:/yolo/yolov5/runs/train/reallPills_200_yolov5s_results/folder/WD.jpg"
# east = "D:/yolo/east/frozen_east_text_detection.pb"
# csv_file = "D:/yolo/pill.csv"
# image = cv2.imread(img, cv2.IMREAD_COLOR)

# try:
#     yolo_label = detect.yolo_run(weights="./yolov5/runs/train/reallPills_200_yolov5s_results/weights/best_2.pt", source= img, save_txt= True)
#     min_x = int(yolo_label[2])
#     max_x = int(yolo_label[4])
#     min_y = int(yolo_label[1])
#     max_y = int(yolo_label[3])

# except:
#     print('error but except!')
    
# finally:
#     print("successfully finish!")

# yolo_roi = image[min_x:max_x, min_y:max_y].copy()

# color_dct = color.color_run(yolo_roi.copy())

# cv2.imwrite('yolo.jpg', yolo_roi)
# height, width = yolo_roi.shape[:2]
# #0.4 적정
# M = cv2.getRotationMatrix2D((width/2, height/2), 0, 0.4)
# yolo_roi = cv2.warpAffine(yolo_roi, M, (width, height))

# text_list = []

# treated_img = pretreat.treat(yolo_roi)
# # cv2.imshow("treat", treated_img)
# east_label = text_detection.east_run(treated_img, east)
# for i in east_label:
#     min_x = i[0]
#     max_x = i[2]
#     min_y = i[1]
#     max_y = i[3]

#     east_roi = yolo_roi[min_y:max_y, min_x:max_x].copy()
#     cv2.imwrite('D:/yolo/deep_text_recognition_benchmark/demo_image/east_roi.png', east_roi)
#     height, width = east_roi.shape[:2]
#     #0.4 적정
#     M = cv2.getRotationMatrix2D((width/2, height/2), 0, 0.4)
#     east_roi_reduction = cv2.warpAffine(east_roi, M, (width, height))
#     # cv2.imshow('east', east_roi)

# try:
#     yolo_ocr = madeOcr.main_use(yolo_roi)
#     text_list.append(yolo_ocr)
# except:
#     print('detect Nothing')
#     yolo_ocr = ''
#     text_list.append(yolo_ocr)

# try:
#     east_ocr = madeOcr.main_use(east_roi_reduction)
#     text_list.append(east_ocr)
# except:
#     print('detect Nothing')
#     east_ocr = ''
#     text_list.append(east_ocr)

# try:
#     keras_yolo_text = keras.keras_run(yolo_roi)
#     text_list.append(keras_yolo_text)
# except:
#     print('detect Nothing')
#     keras_yolo_text = ''
#     text_list.append(keras_yolo_text)

# try:
#     keras_east_text = keras.keras_run(east_roi_reduction)
#     text_list.append(keras_east_text)
# except:
#     print('detect Nothing')
#     keras_east_text = ''
#     text_list.append(keras_east_text)

# bench_mark_text = demo.demo()
# if bench_mark_text == None:
#     bench_mark_text = ''
# text_list.append(bench_mark_text)

# result = []
# for i in text_list:
#     result.append(csv.main(csv_file, color_dct, i))

# for i in result:
#     for j in i:
#         print(j)

app = QApplication(sys.argv)
ex = DCA()
sys.exit(app.exec_())