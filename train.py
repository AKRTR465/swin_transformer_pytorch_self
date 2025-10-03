import os
import argparse
import torch.optim.lr_scheduler as lr_scheduler
import math
import torch
import shutil
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from make_dataset import MyDataset
from model import swin_tiny_patch4_window7_224 as create_model
from train_utils import train_one_epoch, evaluate, set_seed
from build_dataset import read_split_data
from pre_transforms import pre_transform_images

def main(args):

    #设置随机种子
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    #创建模型保存路径
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    
    #创建tensorboard
    tb_writer = SummaryWriter()

    train_images_path, train_images_label, _, _, _ = read_split_data(args.data_path1)
    _, _, val_images_path, val_images_label, _ = read_split_data(args.data_path2)

    NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    #从预处理后的数据集中加载数据，因此只需要转化为张量
    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(NORM_MEAN.squeeze().tolist(), NORM_STD.squeeze().tolist())]),
        "val": transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize(NORM_MEAN.squeeze().tolist(), NORM_STD.squeeze().tolist())])
        }
    
    #创建数据集
    train_dataset = MyDataset(images_path=train_images_path, 
                              images_class=train_images_label, 
                              transform=data_transform["train"])
    val_dataset = MyDataset(images_path=val_images_path, 
                            images_class=val_images_label, 
                            transform=data_transform["val"])
    
    #创建数据加载器
    batch_size = args.batch_size
    #设置线程数
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = batch_size,
                                               shuffle = True,
                                               pin_memory = True,
                                               num_workers = nw,
                                               collate_fn = train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size = batch_size,
                                             shuffle = False,
                                             pin_memory = True,
                                             num_workers = nw,
                                             collate_fn = val_dataset.collate_fn)
    
    #创建模型
    model = create_model(num_classes=args.num_classes).to(device)
    
    #设置迁移学习
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file not found"
        #map_location参数指定模型加载到z指定设备，model则代表只取模型参数
        weights_dict = torch.load(args.weights, map_location = device)["model"]
        for k in list(weights_dict.keys()):
            #删除分类器头，方便后面拼接
            if "head" in k:
                del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    #冻结权重    
    if args.freeze_layers:
        for name, para in model.named_parameters():
            #冻结除分类器以外的层
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print(f"training {name}")
    
    #设置优化器
    #pg作为一个迭代器，遍历所有需要训练的参数，包括权重和偏置等等
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr = args.lr, weight_decay = 5E-2)
    lf = lambda x: (((1 + math.cos(x * math.pi / args.epochs)) / 2) ) * (1 - args.lrf) + args.lrf +0.00002
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda = lf)
    
    acc = 0.

    #开始训练
    for epoch in range(args.epochs):

        #训练
        train_loss, train_acc = train_one_epoch(model = model, 
                                                optimizer = optimizer, 
                                                data_loader = train_loader, 
                                                device = device, 
                                                epoch = epoch, )
        scheduler.step()

        #验证
        val_loss, val_acc = evaluate(model = model, 
                                     data_loader = val_loader, 
                                     device = device, 
                                     epoch = epoch, )
        
        #tensorboard记录
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc > acc:
            acc = val_acc
            #保存最优模型
            torch.save(model.state_dict(), f"./weights/model_{epoch}.pth")

class Args:
    def __init__(self):
        self.device = "cuda"
        #原始数据集的根目录
        self.data_path = "../../data_set/flower_data/flower_photos" 
        #预处理文件保存目录
        self.preprocessed_output_path = "./preprocessed_dataset_output" 
        self.save_preprocessed_data = True
        self.exit_after_preprocessing = True

if __name__ == "__main__":

    if os.path.exists("./preprocessed_dataset_output"):
        shutil.rmtree("./preprocessed_dataset_output")
    args_save = Args()
    args_save.save_preprocessed_data = True
    args_save.exit_after_preprocessing = True
    pre_transform_images(args_save)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=0.004)
    parser.add_argument('--lrf', type=float, default=0.01)

    parser.add_argument('--data-path1', type=str,
                        default="./preprocessed_dataset_output/train")
    parser.add_argument('--data-path2', type=str,
                        default="./preprocessed_dataset_output/val")

    parser.add_argument('--weights', type=str, default='./swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')

    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id')
    parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')

    opt = parser.parse_args()

    main(opt)

    
    


    
    
