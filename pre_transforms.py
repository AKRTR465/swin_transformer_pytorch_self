import os
import torch
from torchvision import transforms, datasets
from PIL import Image
import shutil
import build_dataset


#设置归一化参数, 分别为均值和方差
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def denormalize_tensor_for_image(tensor):
    """
    将归一化后的tensor反归一化到0-1范围内，方便后续转换为 PIL Image
    """
    denormalized_tensor = tensor*NORM_STD + NORM_MEAN
    #确保值在0-1范围内
    denormalized_tensor = torch.clamp(denormalized_tensor, 0, 1)
    
    return denormalized_tensor

def save_transforms_images(images_path, 
                           image_labels, 
                           class_to_index,
                           output_root_dir,
                           subset_name):
    """
    将预处理后的图片保存到指定目录
    """
    """
    创建输出目录
    在指定位置和创建指定名字的目录
    在指定目录下创建子目录，每个子目录对应一个类别（通过build_dataset文件获取）
    """
    os.makedirs(output_root_dir, exist_ok = True)
    subset_output_dir = os.path.join(output_root_dir, subset_name)
    os.makedirs(subset_output_dir, exist_ok = True)

    idx_to_class = {idx : cls_name for cls_name, idx in class_to_index.items()}
    for class_name in class_to_index.keys():
        os.makedirs(os.path.join(subset_output_dir, class_name), exist_ok = True)
    
    to_pil = transforms.ToPILImage()

    print(f"\n pre-processing {subset_name} to: {subset_output_dir} ")
    
    """
    zip将多个对象打包成映射的元组
    """
    current_transform = transform_image(subset_name)
    for i, (path, label_idx) in enumerate(zip(images_path, image_labels)):
        try:
            #将图像强制转换为RGB格式
            img = Image.open(path).convert('RGB')
            transform_tensor = current_transform(img)

            denormalized_tensor = denormalize_tensor_for_image(transform_tensor)
            output_img = to_pil(denormalized_tensor)

            class_name = idx_to_class[label_idx]
            original_filename = os.path.basename(path)
            base_name, ext = os.path.splitext(original_filename)

            filename_to_save = f"{base_name}_{i}{ext}"
            output_path = os.path.join(subset_output_dir, class_name, filename_to_save)
            
            output_img.save(output_path)

            if (i+1) % 100 == 0:
                print(f"saved {i+1}/{len(images_path)} {subset_name}")
        except Exception:
            print(f"failed {path} : {Exception}")
        
            
def transform_image(pos):
    """
    预处理图片
    """
    image_size = 224
    data_transforms = {
        'train': transforms.Compose([transforms.RandomResizedCrop(image_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(NORM_MEAN.squeeze().tolist(), NORM_STD.squeeze().tolist())]),
        "val": transforms.Compose([transforms.Resize(int(image_size * 1.143)),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(NORM_MEAN.squeeze().tolist(), NORM_STD.squeeze().tolist())])
    }
    return data_transforms[pos]

def pre_transform_images(args):

    train_images_path, train_image_labels, val_images_path, val_image_labels, class_to_index = build_dataset.read_split_data(args.data_path)

    if args.save_preprocessed_data:
        if not args.preprocessed_output_path:
            raise ValueError("please make sure to provide the preprocessed output path")
        
        save_transforms_images(train_images_path,
                               train_image_labels,
                               class_to_index,
                               args.preprocessed_output_path,
                               "train")
        save_transforms_images(val_images_path,
                               val_image_labels,
                               class_to_index,
                               args.preprocessed_output_path,
                               "val")
        
class Args:
    def __init__(self):
        self.device = "cuda"
        self.data_path = "../../data_set/flower_data/flower_photos"
        self.preprocessed_output_path = "./preprocessed_dataset_output"
        self.save_preprocessed_data = True
        self.exit_after_preprocessing = True

if __name__ == '__main__':
    # 清理旧的虚拟数据和预处理输出，以便重新运行演示
    if os.path.exists("./preprocessed_dataset_output"):
        shutil.rmtree("./preprocessed_dataset_output")
        
    args_save = Args()
    args_save.save_preprocessed_data = True
    args_save.exit_after_preprocessing = True # 保存后退出
    pre_transform_images(args_save)
