import os
import re
import math
from scipy.stats import pearsonr
import imutils
import cv2
import numpy as np
import pandas as pd
from config import opt
import copy
import matplotlib.pyplot as plt
from tqdm import tqdm

img_res = 0.708 * 0.708  # 单位 mm^2   #输入Confocal图像分辨率，从ZEN软件里查看。这个与放大倍数/视野大小有关，每次计算必须确保此值正确！！！！
pix_res = 1.024 * 1.024  # 确保每次拍得的图像分辨率是1024*1024.否则计算结果将有误！！！！！！！
font = cv2.FONT_HERSHEY_SIMPLEX
radius_1 = 100  # 70 微米
radius_2 = 200  # 140 微米

def mkdir(path):
    '''make dir'''
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2) * ((img_res / pix_res) ** 0.5)


def num_to_BGR(nums):
  r, g, b,_ = plt.cm.hot(nums)
  return (int(255*b),int(255*g),int(255*r))


def calculate(path, type = "Simple", use_magenta = False):
    # """
    # 定量共聚焦图像中的单一血管形态/渗透性质指标，
    # 路径示例：path = r"C:\Users\ZMS\Desktop\Confocal\test\aSMA-3LL\Image 43"
    # 文件夹中包括血管通道图像，NPs通道图像，细胞核通道图像，其他（c图像和对应的分割结果
    # type = "Simple, Medium, Complex"
    # Simple:包含简单参数
    # Medium:包含中等参数
    # Complex:包含复杂参数
    # use_magenta = "True,False"是否使用其他通道(SMA,VEGFR,Macrophage,etc)
    # """

    test_num = path.split("\\")[-1]
    #读取共聚焦图像
    image_red = cv2.imread(path + "/" + test_num + opt.FTn_images_name)
    print(path + "/" + test_num + opt.FTn_images_name)
    image_green = cv2.imread(path + "/" + test_num + opt.Vessel_images_name)
    r_b, r_g, r_r = cv2.split(image_red)
    g_b, g_g, g_r = cv2.split(image_green)
    #读取分割图像
    pre_red = cv2.imread(path + '/predict_1.1.png')
    pre_green = cv2.imread(path + '/predict_2.1.png')
    _, pre_red = cv2.threshold(cv2.cvtColor(pre_red.copy(), cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)
    _, pre_green = cv2.threshold(cv2.cvtColor(pre_green.copy(), cv2.COLOR_BGR2GRAY), 10, 255, cv2.THRESH_BINARY)

    #读取细胞核图像和X通道图像
    if use_magenta != False:

        #image_nuclei = cv2.imread(path + "/" + test_num + opt.Nuclei_images_name)
        image_magenta = cv2.imread(path + "/" + test_num + opt.Magenta_images_name)
        pre_megenta = cv2.imread(path + '/predict_3.1.png')



    #寻找血管&NPs渗透的对应关系，将其体现融合在一起
    pre_pen = pre_red.copy()
    pre_pen[ pre_green > 0 ] = 255
    pre_penetrate = pre_pen.copy()

    #对于每个血管&NPs渗透的对应关系进行识别
    pre_contours_penetrate = cv2.findContours(pre_penetrate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pre_contours_penetrate = imutils.grab_contours(pre_contours_penetrate)

    area_vess_list = []  # 1.  每个血管面积
    area_ftn_list = []   # 2. 每个血管渗透面积
    PR = []              # 3. 每个血管渗透面积比
    FA = []              # 4. 每个血管渗透量
    TN = []              # 5. 每个血管周边总NPs量
    VP = []              # 6. 每个血管渗透率
    pearson_list = []    # 7. 皮尔森共定位系数

    if type == "Medium":
        perimeter_list = []         # 8.  每个血管周长
        perimeter_ratio_list = []   # 9.  每个血管周长面积比
        aspect_ratio_list = []      # 10. 每个血管横纵比
        locdesnity1_list = []       # 11. 每个血管局部密度1
        locdesnity2_list = []       # 12. 每个血管局部密度2


    elif type == "Complex":
        perimeter_list = []         # 8.   每个血管周长
        perimeter_ratio_list = []   # 9.   每个血管周长面积比
        aspect_ratio_list = []      # 10， 每个血管横纵比
        locdesnity1_list = []       # 11.  每个血管局部密度1
        locdesnity2_list = []       # 12.  每个血管局部密度2

        points = []                 # 0.   每个血管点坐标
        solidity_list = []          # 13.  每个血管凸包面积比
        abs_asp_ratio_list = []     # 14.  每个血管相对横纵比


    if use_magenta == "SMA":
        # blurred_sma = cv2.GaussianBlur(image_magenta, (5, 5), 0)
        # gray_sma = cv2.cvtColor(blurred_sma, cv2.COLOR_BGR2GRAY)
        # _, gray_sma = cv2.threshold(gray_sma, 40, 255, cv2.THRESH_BINARY)  # 黑白二值化
        # cv2.imwrite(path + '/predict 3.1.png', gray_sma)
        SMA_rate_list = []
        gray_sma = cv2.cvtColor(pre_megenta, cv2.COLOR_BGR2GRAY)
        _, gray_sma = cv2.threshold(gray_sma, 40, 255, cv2.THRESH_BINARY)


    # 对于每个血管&NPs渗透的对应关系进行迭代
    for cnt in tqdm(pre_contours_penetrate):
        if cv2.arcLength(cnt, True) > 0:
            mask_pen = np.zeros_like(pre_pen)
            mask_vess = np.zeros_like(pre_pen)
            mask_ftn = np.zeros_like(pre_pen)

            cv2.drawContours(mask_pen, [cnt], -1, 255, thickness=cv2.FILLED)
            #识别每个（组）血管
            mask_vess[mask_pen * pre_green > 0] = 255
            # 识别每个NPs渗透
            mask_ftn[mask_pen * pre_red > 0] = 255
            area_vess = np.sum(mask_vess / 255) * img_res / pix_res
            area_ftn = np.sum(mask_ftn / 255) * img_res / pix_res

            if area_vess > 50:
                contours_singel_vess, _ = cv2.findContours(mask_vess.copy(), cv2.RETR_TREE,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                totalarea_vess = 0
                for c1 in contours_singel_vess:
                    if cv2.contourArea(c1) >= 50:
                        totalarea_vess = totalarea_vess + cv2.contourArea(c1)

                for c1 in contours_singel_vess:
                    if cv2.contourArea(c1) >= 50:
                        cnt_area = cv2.contourArea(c1) * img_res / pix_res  # 血管轮廓面积
                        i = totalarea_vess / cv2.contourArea(c1)
                        mask_i = np.zeros_like(pre_pen)
                        cv2.drawContours(mask_i, [c1], -1, 255, thickness=cv2.FILLED)

                        if use_magenta == "SMA":
                            sma = gray_sma/255 * mask_i
                            sma = np.clip(sma, 0, 255)
                            rate = np.sum(sma / 255) / np.sum(mask_i / 255)

                        ftn_in_vess = np.sum(image_red[mask_i * pre_green > 0])
                        ftn_pen_i = np.sum(image_red[mask_pen * pre_red > 0]) / i
                        ftn_acc = ftn_pen_i - ftn_in_vess
                        vess_pen_i = ftn_acc / ftn_pen_i
                        if vess_pen_i < 0:
                            vess_pen_i = 0
                            ftn_acc = 0

                        red = r_r * mask_pen / 255
                        green = g_g * mask_pen / 255

                        green_flat = green.flatten()
                        red_flat = red.flatten()
                        pearson, _ = pearsonr(green_flat, red_flat)


                        area_vess_list.append(cnt_area)
                        area_ftn_list.append(area_ftn / i)
                        PR.append(area_ftn / (i * cnt_area))
                        FA.append(ftn_acc)
                        TN.append(ftn_pen_i)
                        VP.append(vess_pen_i)
                        pearson_list.append(pearson)

                        if use_magenta == "SMA":
                            SMA_rate_list.append(rate)


                        if type == "Medium":

                            perimeter = cv2.arcLength(c1, True) * ((img_res / pix_res) ** 0.5)
                            perimeter_ratio = perimeter / cnt_area

                            (x, y), (w, h), angel = cv2.minAreaRect(c1)
                            center_x = int(x + 0.5 * w)
                            center_y = int(y + 0.5 * h)

                            circle_img1 = np.zeros_like(pre_pen, dtype=np.uint8)
                            circle_img2 = np.zeros_like(pre_pen, dtype=np.uint8)
                            cv2.circle(circle_img1, (center_x, center_y), radius_1, 1, thickness=cv2.FILLED)
                            cv2.circle(circle_img2, (center_x, center_y), radius_2, 1, thickness=cv2.FILLED)

                            img_des1 = pre_green * circle_img1
                            img_des2 = pre_green * circle_img2

                            countourofthisvessl_1, _ = cv2.findContours(img_des1, mode=cv2.RETR_TREE,
                                                                        method=cv2.CHAIN_APPROX_SIMPLE)
                            countourofthisvessl_2, _ = cv2.findContours(img_des2, mode=cv2.RETR_TREE,
                                                                        method=cv2.CHAIN_APPROX_SIMPLE)
                            locdesnity_1 = len(countourofthisvessl_1)
                            locdesnity_2 = len(countourofthisvessl_2)

                            if w <= h:  # 求横纵比（0-1分布）
                                aspect_ratio = float(w) / h
                            else:
                                aspect_ratio = float(h) / w

                            perimeter_list.append(perimeter)
                            perimeter_ratio_list.append(perimeter_ratio)
                            aspect_ratio_list.append(aspect_ratio)
                            locdesnity1_list.append(locdesnity_1)
                            locdesnity2_list.append(locdesnity_2)

                        if type == "Complex":

                            perimeter = cv2.arcLength(c1, True) * ((img_res / pix_res) ** 0.5)
                            perimeter_ratio = perimeter / cnt_area

                            (x, y), (w, h), angel = cv2.minAreaRect(c1)
                            center_x = int(x + 0.5 * w)
                            center_y = int(y + 0.5 * h)

                            circle_img1 = np.zeros_like(pre_pen, dtype=np.uint8)
                            circle_img2 = np.zeros_like(pre_pen, dtype=np.uint8)
                            cv2.circle(circle_img1, (center_x, center_y), radius_1, 1, thickness=cv2.FILLED)
                            cv2.circle(circle_img2, (center_x, center_y), radius_2, 1, thickness=cv2.FILLED)

                            img_des1 = pre_green * circle_img1
                            img_des2 = pre_green * circle_img2

                            countourofthisvessl_1, _ = cv2.findContours(img_des1, mode=cv2.RETR_TREE,
                                                                        method=cv2.CHAIN_APPROX_SIMPLE)
                            countourofthisvessl_2, _ = cv2.findContours(img_des2, mode=cv2.RETR_TREE,
                                                                        method=cv2.CHAIN_APPROX_SIMPLE)
                            locdesnity_1 = len(countourofthisvessl_1)
                            locdesnity_2 = len(countourofthisvessl_2)

                            if w <= h:  # 求横纵比（0-1分布）
                                aspect_ratio = float(w) / h
                            else:
                                aspect_ratio = float(h) / w

                            hull_points = cv2.convexHull(c1)  # 获取血管凸包点
                            hull_area = cv2.contourArea(hull_points) * img_res / pix_res  # 血管凸包廓面积
                            if hull_area == 0:
                                solidity = 1
                            else:
                                solidity = float(cnt_area) / hull_area  # 血管凸包面积比

                            abs_asp_ratio = float(perimeter_ratio) * solidity  # 相对横纵比

                            perimeter_list.append(perimeter)
                            perimeter_ratio_list.append(perimeter_ratio)
                            aspect_ratio_list.append(aspect_ratio)
                            locdesnity1_list.append(locdesnity_1)
                            locdesnity2_list.append(locdesnity_2)
                            points.append([c1])
                            solidity_list.append(solidity)
                            abs_asp_ratio_list.append(abs_asp_ratio)

                        if use_magenta == True:
                            pass

    if type == "Simple":
        dataDF_pre = pd.concat([
            pd.DataFrame({'1.Area of each BV': area_vess_list}),
            pd.DataFrame({'2.PA of each BV': area_ftn_list}),
            pd.DataFrame({'3.PR of each BV': PR}),
            pd.DataFrame({'4.FA of each BV': FA}),
            pd.DataFrame({'5.Total FTn of each BV': TN}),
            pd.DataFrame({'6.VP of each BV': VP}),
            pd.DataFrame({'7.Pearsons of each BV': pearson_list})
        ],
            axis=1)

    if use_magenta == "SMA" and type == "Simple":
        dataDF_pre = pd.concat([
            pd.DataFrame({'1.Area of each BV': area_vess_list}),
            pd.DataFrame({'2.PA of each BV': area_ftn_list}),
            pd.DataFrame({'3.PR of each BV': PR}),
            pd.DataFrame({'4.FA of each BV': FA}),
            pd.DataFrame({'5.Total FTn of each BV': TN}),
            pd.DataFrame({'6.VP of each BV': VP}),
            pd.DataFrame({'7.Pearsons of each BV': pearson_list}),
            pd.DataFrame({'8.SMA cover rate of each BV': SMA_rate_list})

        ],
            axis=1)



    elif type == "Medium":
        dataDF_pre = pd.concat([
            pd.DataFrame({'1.Area of each BV': area_vess_list}),
            pd.DataFrame({'2.PA of each BV': area_ftn_list}),
            pd.DataFrame({'3.PR of each BV': PR}),
            pd.DataFrame({'4.FA of each BV': FA}),
            pd.DataFrame({'5.Total FTn of each BV': TN}),
            pd.DataFrame({'6.VP of each BV': VP}),
            pd.DataFrame({'7.Pearsons of each BV': pearson_list}),
            pd.DataFrame({'8.Perimeter of each BV': perimeter_list}),
            pd.DataFrame({'9.Perimeter area ratio of each BV': perimeter_ratio_list}),
            pd.DataFrame({'10.Aspect of each BV': aspect_ratio_list}),
            pd.DataFrame({'11.Local density_1 of each BV': locdesnity1_list}),
            pd.DataFrame({'12.Local density_2 of each BV': locdesnity2_list})

        ],
            axis=1)

    elif type == "Complex":
        dataDF_pre = pd.concat([
            pd.DataFrame({'0.Coordinates of each BV': points}),
            pd.DataFrame({'1.Area of each BV': area_vess_list}),
            pd.DataFrame({'2.PA of each BV': area_ftn_list}),
            pd.DataFrame({'3.PR of each BV': PR}),
            pd.DataFrame({'4.FA of each BV': FA}),
            pd.DataFrame({'5.Total FTn of each BV': TN}),
            pd.DataFrame({'6.VP of each BV': VP}),
            pd.DataFrame({'7.Pearsons of each BV': pearson_list}),
            pd.DataFrame({'8.Perimeter of each BV': perimeter_list}),
            pd.DataFrame({'9.Perimeter area ratio of each BV': perimeter_ratio_list}),
            pd.DataFrame({'10.Aspect of each BV': aspect_ratio_list}),
            pd.DataFrame({'11.Local density_1 of each BV': locdesnity1_list}),
            pd.DataFrame({'12.Local density_2 of each BV': locdesnity2_list}),
            pd.DataFrame({'13.Solidity of each BV': solidity_list}),
            pd.DataFrame({'14.Abs_Aspect of each BV': abs_asp_ratio_list})

        ],
            axis=1)
        dataDF_pre.to_json(path + "/pre_{}.json".format(test_num), orient='records', lines=True)

    dataDF_pre.to_csv(path + "/pre_{}.csv".format(test_num), mode='a',index=False, encoding='utf_8_sig')

if __name__ == "__main__":

    root = r"C:\Users\ZMS\Desktop\Confocal\test"

    for tumor in os.listdir(root):
        tumor_path = os.path.join(root,tumor)
        for img in os.listdir(tumor_path):
            img_path = os.path.join(tumor_path,img)
            calculate(img_path, use_magenta = "SMA")
            print(img_path,"已完成")

