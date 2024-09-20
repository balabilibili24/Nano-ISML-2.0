import os
import cv2
import numpy as np


data_path = r'C:\Users\Huang Lab\Desktop\gauss'  #存放位置
Images_names = os.listdir(data_path)

for Images in Images_names:
    Image_path = data_path + "/" + Images
    img = cv2.imread(Image_path)
    dst = cv2.GaussianBlur(img,(0,0), sigmaX=0.8, sigmaY=0.8)
    cv2.imwrite(data_path + "/" + f"{Images}-1.png", dst, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])






