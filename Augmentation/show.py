import xml.dom.minidom
import cv2
import os

"""
该脚本用于目标框可视化
IMAGE_INPUT_PATH：输入图片路径
XML_INPUT_PATH：输入标记框路径
IMAGE_OUTPUT_PATH：生成可视化图片路径
"""
IMAGE_INPUT_PATH = 'VOC2007/JPEGImages_after'
XML_INPUT_PATH = 'VOC2007/Annotations_after'
IMAGE_OUTPUT_PATH = 'VOC2007/show_output'
imglist = os.listdir(IMAGE_INPUT_PATH)
xmllist = os.listdir(XML_INPUT_PATH)

for i in range(len(imglist)):
    # 每个图像全路径
    image_input_fullname = IMAGE_INPUT_PATH + '/' + imglist[i]
    xml_input_fullname = XML_INPUT_PATH + '/' + xmllist[i]
    image_output_fullname = IMAGE_OUTPUT_PATH + '/' + imglist[i]

    img = cv2.imread(image_input_fullname)

    dom = xml.dom.minidom.parse(xml_input_fullname)
    root = dom.documentElement

    # 读取标注目标框
    objects = root.getElementsByTagName("bndbox")

    for object in objects:
        xmin = object.getElementsByTagName("xmin")
        xmin_data = int(float(xmin[0].firstChild.data))
        ymin = object.getElementsByTagName("ymin")
        ymin_data = int(float(ymin[0].firstChild.data))
        xmax = object.getElementsByTagName("xmax")
        xmax_data = int(float(xmax[0].firstChild.data))
        ymax = object.getElementsByTagName("ymax")
        ymax_data = int(float(ymax[0].firstChild.data))

        # 显示缩放后的目标框
        # xmin, ymin, xmax, ymax分别为xml读取的坐标信息
        left_top = (int(xmin_data), int(ymin_data))
        right_down = (int(xmax_data), int(ymax_data))
        cv2.rectangle(img, left_top, right_down, (255, 0, 0), 1)
        """
        # 直接查看生成结果图 
        cv2.imshow('show', img)
        cv2.waitKey(0)
        """

    cv2.imwrite(image_output_fullname, img)
