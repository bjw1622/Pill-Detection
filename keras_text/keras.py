# 맨 위에꺼 pip 해주세용
# # To install from master
# pip install git+https://github.com/faustomorales/keras-ocr.git#egg=keras-ocr

# # To install from PyPi
# pip install keras-ocr

import keras_ocr
import matplotlib.pyplot as plt
import numpy as np


def keras_run(img):
  pipeline = keras_ocr.pipeline.Pipeline()
  img = [np.array(img)]
  # images = [
  #     keras_ocr.tools.read(img) for img in ['D:\\yolo\\yolov5\\runs\\train\\reallPills_200_yolov5s_results\\kss.jpg'
  #                                         #   ,
  #                                         #   'C:\\Users\\bjw16\\Desktop\\VSCODE\\python\\OCR\\image\\kss.jpg'
  #     ]
  # ]
  # len(images)
  # plt.figure(figsize = (10,20))
  # plt.imshow(images[0])
  # plt.figure(figsize = (10,20))
  # plt.imshow(images[1])
  prediction_groups = pipeline.recognize(img)
  text = ''
  for i in prediction_groups:
    for j in i:
      text += j[0]
  # print(text)
  return text
  # fig, axs = plt.subplots(nrows=len(images), figsize=(10, 20))
  # for ax, image, predictions in zip(axs, images, prediction_groups):
  #     keras_ocr.tools.drawAnnotations(image=image, 
  #                                     predictions=predictions, 
  #                                     ax=ax)
  #     print(predictions)