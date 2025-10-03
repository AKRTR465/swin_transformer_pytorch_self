import numpy as np
import sys
import random
from tqdm import tqdm
import torch
import torch.nn as nn

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    一轮内的训练过程
    """
    #设定为训练模式
    model.train()
    loss_function = nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    #清空梯度
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file = sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        #获取批次大小
        sample_num += images.shape[0]
        #进行前向传播
        pred = model(images.to(device))
        #得到预测结果
        pred_classes = torch.max(pred, dim=1)[1]
        #判断是否正确，并计算正确的个数
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        #进行反向传播
        loss.backward()
        #累加损失，同时与训练参数分离
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        
        if not torch.isfinite(loss):
            print("Loss is {}, stopping training".format(loss.item()), file=sys.stderr)
            sys.exit(1)

        #更新参数,根据反向传播的梯度进行优化
        optimizer.step()
        #清空梯度
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    """
    进行val，通过torch.no_grad()装饰器禁用梯度计算
    """
    #设定为验证模式
    model.eval()
    
    loss_function = nn.CrossEntropyLoss()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file = sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        #获取批次大小
        sample_num += images.shape[0]
        #进行前向传播
        pred = model(images.to(device))
        #得到预测结果
        pred_classes = torch.max(pred, dim=1)[1]
        #判断是否正确，并计算正确的个数
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

def set_seed(seed):
    """
    固定所有可能的随机种子以确保实验的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")

if __name__ == '__main__':
    set_seed(42)


