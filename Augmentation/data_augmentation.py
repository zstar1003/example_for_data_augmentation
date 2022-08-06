import xml.etree.ElementTree as ET
import pickle
import os
from os import getcwd
import numpy as np
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)


# 读取出图像中的目标框
def read_xml_annotation(root, image_id):
    in_file = open(os.path.join(root, image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    bndboxlist = []

    for object in root.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        xmin = int(bndbox.find('xmin').text)
        xmax = int(bndbox.find('xmax').text)
        ymin = int(bndbox.find('ymin').text)
        ymax = int(bndbox.find('ymax').text)
        # print(xmin,ymin,xmax,ymax)
        bndboxlist.append([xmin, ymin, xmax, ymax])
        # print(bndboxlist)

    return bndboxlist  # 以多维数组的形式保存


def change_xml_annotation(root, image_id, new_target):
    new_xmin = new_target[0]
    new_ymin = new_target[1]
    new_xmax = new_target[2]
    new_ymax = new_target[3]

    in_file = open(os.path.join(root, str(image_id) + '.xml'))
    tree = ET.parse(in_file)
    xmlroot = tree.getroot()
    object = xmlroot.find('object')
    bndbox = object.find('bndbox')
    xmin = bndbox.find('xmin')
    xmin.text = str(new_xmin)
    ymin = bndbox.find('ymin')
    ymin.text = str(new_ymin)
    xmax = bndbox.find('xmax')
    xmax.text = str(new_xmax)
    ymax = bndbox.find('ymax')
    ymax.text = str(new_ymax)
    tree.write(os.path.join(root, str(image_id) + "_aug" + '.xml'))


def change_xml_list_annotation(root, image_id, new_target, saveroot, id):
    in_file = open(os.path.join(root, str(image_id) + '.xml'))  # 读取原来的xml文件
    tree = ET.parse(in_file)  # 读取xml文件
    xmlroot = tree.getroot()
    index = 0
    # 将bbox中原来的坐标值换成新生成的坐标值
    for object in xmlroot.findall('object'):  # 找到root节点下的所有country节点
        bndbox = object.find('bndbox')  # 子节点下节点rank的值

        # 注意new_target原本保存为高维数组
        new_xmin = new_target[index][0]
        new_ymin = new_target[index][1]
        new_xmax = new_target[index][2]
        new_ymax = new_target[index][3]

        xmin = bndbox.find('xmin')
        xmin.text = str(new_xmin)
        ymin = bndbox.find('ymin')
        ymin.text = str(new_ymin)
        xmax = bndbox.find('xmax')
        xmax.text = str(new_xmax)
        ymax = bndbox.find('ymax')
        ymax.text = str(new_ymax)

        index = index + 1

    tree.write(os.path.join(saveroot, str(image_id) + "_aug_" + str(id) + '.xml'))
    # tree.write(os.path.join(saveroot, str(image_id) + "_baseaug_" + '.xml'))


# 处理文件
def mkdir(path):
    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False


if __name__ == "__main__":

    IMG_DIR = r"C:\Users\xy\Desktop\Augmentation\VOC2007\JPEGImages"
    XML_DIR = r"C:\Users\xy\Desktop\Augmentation\VOC2007\Annotations"

    # 存储增强后的影像文件夹路径
    AUG_IMG_DIR = r"C:\Users\xy\Desktop\Augmentation\VOC2007\JPEGImages_after"
    mkdir(AUG_IMG_DIR)
    # 存储增强后的XML文件夹路径
    AUG_XML_DIR = r"C:\Users\xy\Desktop\Augmentation\VOC2007\Annotations_after"
    mkdir(AUG_XML_DIR)

    AUGLOOP = 2  # 每张影像增强的数量

    boxes_img_aug_list = []
    new_bndbox = []
    new_bndbox_list = []

    # 影像增强
    seq = iaa.Sequential([
        iaa.Flipud(0.5),  # 对50%的图像做上下翻转
        iaa.Fliplr(0.5),  # 对50%的图像做左右翻转
        # iaa.Multiply((1.2, 1.5)),  # 像素乘上1.2或者1.5之间的数字
        # iaa.GaussianBlur(sigma=(0, 3.0)),  # 使用高斯模糊
        iaa.WithColorspace(to_colorspace="HSV", from_colorspace="RGB",
                           children=iaa.WithChannels(0, iaa.Add(10))),  # 先将图片从RGB变换到HSV,然后将H值增加10,然后再变换回RGB
        iaa.Dropout((0.01, 0.1), per_channel=0.5),  # 将1%到10%的像素设置为黑色
        iaa.Affine(
            # translate_px={"x": 15, "y": 15},  # 平移
            scale=(0.8, 0.95),  # 图像缩放为80%到95%之间
            rotate=(-30, 30)  # 旋转±30度之间
        )  # 仿射变换
    ])

    # 得到当前运行的目录和目录当中的文件，其中sub_folders可以为空
    for root, sub_folders, files in os.walk(XML_DIR):
        # 遍历每一张图片
        for name in files:
            bndbox = read_xml_annotation(XML_DIR, name)
            for epoch in range(AUGLOOP):
                seq_det = seq.to_deterministic()  # 固定变换序列,之后就可以先变换图像然后变换关键点,这样可以保证两次的变换完全相同

                # 读取图片
                img = Image.open(os.path.join(IMG_DIR, name[:-4] + '.jpg'))
                img = np.array(img)

                # bndbox 坐标增强，依次处理所有的bbox
                for i in range(len(bndbox)):
                    bbs = ia.BoundingBoxesOnImage([
                        ia.BoundingBox(x1=bndbox[i][0], y1=bndbox[i][1], x2=bndbox[i][2], y2=bndbox[i][3]),
                    ], shape=img.shape)

                    bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                    boxes_img_aug_list.append(bbs_aug)

                    # new_bndbox_list:[[x1,y1,x2,y2],...[],[]]
                    new_bndbox_list.append([int(bbs_aug.bounding_boxes[0].x1),
                                            int(bbs_aug.bounding_boxes[0].y1),
                                            int(bbs_aug.bounding_boxes[0].x2),
                                            int(bbs_aug.bounding_boxes[0].y2)])
                # 存储变化后的图片
                image_aug = seq_det.augment_images([img])[0]
                path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                # path = os.path.join(AUG_IMG_DIR, str(name[:-4]) + "_baseaug_" + '.jpg')
                # image_auged = bbs.draw_on_image(image_aug, thickness=0)
                Image.fromarray(image_aug).save(path)

                # 存储变化后的XML
                change_xml_list_annotation(XML_DIR, name[:-4], new_bndbox_list, AUG_XML_DIR, epoch)
                # print(str(name[:-4]) + "_aug_" + str(epoch) + '.jpg')
                new_bndbox_list = []

