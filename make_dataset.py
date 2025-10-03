from PIL import Image
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    自定义数据集
    """

    def __init__(self, images_path: list, images_class: list, transform = None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
    
    #返回数据集的总长度
    def __len__(self):
        return len(self.images_path)
    
    #返回数据集中第index个样本的图像和标签
    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError(f"image {self.images_path[item]} is not RGB mode")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label
    
    #创建静态方法，将同一批数据整理为一个batch
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        #在batch维度上拼接图像
        images = torch.stack(images, dim = 0)
        #将标签转换为tensor
        labels = torch.as_tensor(labels)
        return images, labels