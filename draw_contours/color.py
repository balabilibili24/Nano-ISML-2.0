import cv2

img_path = r"C:\Users\Huang Lab\Desktop\merge1"


# 读取图像并转换为 BGR 格式
img = cv2.imread(img_path + "\merge.png")
#img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# 获取图像的行数和列数
rows, cols = img.shape[:2]

# 遍历每个像素点，判断是否为灰色（122, 122, 122）
for i in range(rows):
    for j in range(cols):
        # 获取当前像素点的颜色值
        b, g, r = img[i, j]
        # 如果是灰色，则替换为白色（255, 255, 255）
        if b == 255 and g == 255 and r == 255:
            img[i, j] = (122, 122, 122)

# 显示或保存替换后的图像
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(img_path +"/result.jpg", img)



