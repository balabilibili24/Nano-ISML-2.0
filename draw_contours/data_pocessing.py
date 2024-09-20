import cv2
import numpy as np
import os

def tianchong(path,rename_path):
    # 读取原始图像（假设是灰度图像，轮廓为白色，背景为黑色）
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # 二值化图像（将图像转换为黑白，0为黑，255为白）
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个与原始图像相同大小的空白图像用于绘制填充轮廓
    filled_image = np.zeros_like(image)

    # 填充所有找到的轮廓，颜色设置为255（白色）
    cv2.drawContours(filled_image, contours, -1, 255, thickness=cv2.FILLED)

    # 保存结果图像
    cv2.imwrite(rename_path, filled_image)


if __name__ == "__main__":
    path = r"C:\Users\ZMS\Desktop\aSMA-0912"
    for tumor in os.listdir(path):
        img_paths = os.path.join(path,tumor)
        for img_num in os.listdir(img_paths):

            path_c1 = img_paths + "/" + img_num + "/mask {}_c1.png".format(img_num)
            rename_path_c1 = img_paths + "/" + img_num + "/predict_1.1.png"
            tianchong(path_c1,rename_path_c1)
            path_c2 = img_paths + "/" +img_num + "/mask {}_c2.png".format(img_num)
            rename_path_c2 = img_paths + "/" + img_num + "/predict_2.1.png"
            tianchong(path_c2, rename_path_c2)
            path_c3 = img_paths + "/" + img_num + "/mask {}_c3.png".format(img_num)
            rename_path_c3 = img_paths + "/" + img_num + "/predict_3.1.png"
            tianchong(path_c3, rename_path_c3)

            print( path_c3)










