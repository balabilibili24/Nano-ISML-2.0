import cv2
import os
import numpy as np

def recognize_sma(path):

    img_path = path + "/Image 50_c3.png"

    img = cv2.imread(img_path)

    blurred_img = cv2.GaussianBlur(img, (5, 5), 0)

    gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)

    _, gray_img = cv2.threshold(gray_img,40, 255, cv2.THRESH_BINARY)  # 黑白二值化

    # gray_img = cv2.Canny(gray_img, 50, 150)

    contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = img.copy()
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), -1)  # 绘制绿色轮廓


    cv2.imwrite(path +'/Original Image.png', img)
    cv2.imwrite(path +'/gray Image.png', gray_img)
    cv2.imwrite(path + '/Blurred Image.png', blurred_img)
    # cv2.imwrite(path +'/Edges.png', edges)
    cv2.imwrite(path +'/Contours.png', contour_image)



if __name__ =="__main__":

    path = r"C:\Users\ZMS\Desktop\Confocal\test\aSMA-3LL\50"

    recognize_sma(path)

