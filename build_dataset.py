import os
import random
import json

def read_split_data(root, val_rate = 0.2):
    
    #设置种子
    random.seed(0)
    assert os.path.exists(root), "data root {} not exists".format(root)

    #遍历文件夹，一个文件夹对应一个类别
    """
    listdir遍历root路径下的所有文件，并返回一个列表
    os.path.join()用于将root和遍历的cla组合成一个完整路径
    os.path.isdir()用于判断是否为文件夹
    """
    label = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    #排序，保证label的顺序一致,sort()为升序排列
    label.sort()
    #生成类别冰城以及对应的数字索引，写入json文件
    """
    enumerate()对label列表中的元素进行解包，返回索引和对应到的元素，构成字典
    dict()将字典中的键值对进行转换，key为元素，val为索引
    json.dumps()将字典转换为json格式的字符串
    然后再次对调变为（索引，元素）的键值对，写入json文件，
    indent = 4使得输出的 JSON 字符串具有4个空格的缩进
    """
    label_indices = dict((k, v) for v, k in enumerate(label))
    json_str = json.dumps(dict((val, key) for key, val in label_indices.items()), indent = 4)
    with open("label_indices.json", "w") as json_file:
        json_file.write(json_str)
    
    #存放图片路径和标签
    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []

    #指出支持的文件后缀类型
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    for cla in label:
        #获取指定类型中所有图片的路径
        cla_path = os.path.join(root, cla)
        """
        拼接获得图片的路径
        splitext()用于分离文件名与扩展名 ('image1', '.jpg')
        [-1]提取后缀进行匹配
        """
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        images_class = label_indices[cla]
        every_class_num.append(len(images))
        """
        按比例随机挑取验证集
        random.sample()会按计算的条数从中随机选取这些条数的图片的路径用于构建验证集
        在这里的挑选是无放回的，val_path实际已经完成了val和train的划分
        """
        val_path = random.sample(images, int(len(images) * val_rate))
        
        #将图片路径和标签分别存入对应的列表中
        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(images_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(images_class)
        
        #打印数据集信息
    print(f"{sum(every_class_num)} images were found in the dataset")
    print(f"{len(train_images_label)} images were found in the training set")
    print(f"{len(val_images_label)} images were found in the validation set")
        
    #检测数据集是否为可训练的数据集
    assert len(train_images_label) > 0, "No training images were found in the dataset"
    assert len(val_images_label) > 0, "No validation images were found in the dataset"

    return train_images_path, train_images_label, val_images_path, val_images_label, label_indices

