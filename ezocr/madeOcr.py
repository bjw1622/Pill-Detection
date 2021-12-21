import easyocr
import cv2

def main_use(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    equ = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(equ, (5, 5), 1)

    reader = easyocr.Reader(['en'])
    result = reader.readtext(equ)

    for i in range(len(result)):
        last = result[i][-2]

    return last