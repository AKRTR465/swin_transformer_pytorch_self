import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

from typing import Optional

#主模型
class SwinTransformer(nn.Module):
    #定义模型参数并继承module块
    def __init__(self,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 num_classes: int = 1000,
                 embed_dim :int = 96,
                 depths = (2, 2, 6, 2),
                 num_heads = (3, 6, 12, 24),
                 window_size : int = 7,
                 mlp_ratio : float = 4.,
                 qkv_bias : bool = True,
                 drop_rate : float = 0.,
                 attn_drop_rate : float = 0.,
                 drop_path_rate : float = 0.1,
                 norm_layer = nn.LayerNorm,
                 patch_norm : bool = True,
                 use_checkpoint : bool = False,
                 **kwargs):
        super(SwinTransformer, self).__init__()
        
        #声明
        self.patch_size = patch_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        """
        patch size为对原图划分的的patch的大小，
        num_layers为模型的层数，
        embed_dim为第一个块的输出维度，通过num_features得到每个块的输出维度，
        patch_norm为是否对patch进行归一化，
        mlp_ratio为mlp的隐藏层通道数与输入通道数的比值，即隐层放大的通道倍率。
        """

        #第一步，将输入图像进行下采样至patch_size大小方便后续进行处理
        self.patch_embed = PatchEmbed(patch_size = patch_size,
                                      in_c = in_chans,
                                      embed_dim = embed_dim,
                                      norm_layer = norm_layer if self.patch_norm else None)
        
        #设置神经元随机关闭
        self.pos_drop = nn.Dropout(p = drop_rate)

        #第二步，构建随机深度框架
        """
        此处的随机路径被丢弃的概率随着深度的增加而逐渐增加，因此要构建一个等差数列列表，
        此处使用torch.linspace函数生成等差数列：torch.linspace(start, end, steps)
        丢弃概率随网络深度的增加而增加
        """
        dpr  = [x.item( ) for x in torch.linspace(0, drop_path_rate, sum(depths))]

        #第三步，构建主层结构
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            layers = BasicLayer(dim = int(embed_dim * 2 ** i_layer),
                                depth = depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size = window_size,
                                mlp_ratio = mlp_ratio,
                                qkv_bias = qkv_bias,
                                drop = drop_rate,
                                attn_drop = attn_drop_rate,
                                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer+1])],
                                norm_layer = norm_layer,
                                downsample = PatchMerging if (i_layer < self.num_layers - 1 ) else None,
                                use_checkpoint = use_checkpoint)
            self.layers.append(layers)
        
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features,num_classes) if num_classes > 0 else nn.Identity()

        #初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    #构建正向传播网络
    def forward(self, x):
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)

        for layer in self.layers:
            x, H, W = layer(x, H, W)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.head(x)
        
        return x

class PatchEmbed(nn.Module):
    """
    将图像划分为不重叠的patch，大小为patch_size*patch_size，
    方便后续进行处理。
    """
    def __init__(self,
                 patch_size : int = 4,
                 in_c : int = 3,
                 embed_dim : int = 96,
                 norm_layer = None):
        super(PatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_c = in_c
        self.embed_dim = embed_dim
        """
        self.proj为卷积层，用embed_dim的通道数进行卷积，
        卷积核大小为patch_size*patch_size，
        步长为patch_size，
        输出为[B, embed_dim, H/patch_size, W/patch_size]
        """
        self.proj = nn.Conv2d(in_c,
                              embed_dim,
                              kernel_size = patch_size,
                              stride = patch_size)
        """
        归一化，默认为不进行归一化（Identity）
        传入指定参数可进行归一化
        """
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    
    #构建正向传播网络
    def forward(self, x):
        _, _, H, W = x.shape
        """
        为了确保图像能够被patch_size整除，先进行padding填充至可被整除
        """
        test_pad = (H % self.patch_size[0] != 0) or (W % self.patch_size[1]!= 0)
        if test_pad:
            #得到整除后“还差”多少元素能够再被整除，将这些数目的零元素填充进原数组
            pad_h = self.patch_size[0] - H % self.patch_size[0]
            pad_w = self.patch_size[1] - W % self.patch_size[1]
            """
            F.pad为逆向填充，因为（B,C,H,W）会被反向为（B,W,H,C）
            所以为（W_left, W_right, H_top, H_bottom, C_front, C_bottom）
            遵循从左到右，从上到下，从前到后，逆向填充
            """
            #填充右方和下方是因为卷积操作是从左到右从上往下
            x = F.pad(0, pad_h, 0, pad_w, 0, 0)
        
        #进行下采样
        x = self.proj(x)
        _, _, H, W = x.shape
        #flatten将H,W展平为HW，然后将C与HW互换
        #[B, C, H, W]->[B, C, HW]->[B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        #进行归一化,此处默认不进行
        x = self.norm(x)
        return x, H, W
    
class BasicLayer(nn.Module):
    def __init__(self,
                 dim : int,
                 depth : int,
                 num_heads : int,
                 window_size : int,
                 mlp_ratio : float = 4.,
                 qkv_bias : bool = True,
                 drop : float = 0.,
                 attn_drop : float = 0.,
                 drop_path : float = 0.,
                 norm_layer : nn.Module = nn.LayerNorm,
                 downsample : Optional[nn.Module] = None,
                 use_checkpoint : bool = False):
        super(BasicLayer, self).__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        #这里的shift_size为移动窗口的长度，为窗口长度的一半，这里是swin_transformer的特点
        self.shift_size = window_size // 2

        #构建主体部分
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim = dim,
                                 num_heads = num_heads,
                                 window_size = window_size,
                                 shift_size = 0 if (i % 2 == 0) else self.shift_size,
                                 mlp_ratio = mlp_ratio,
                                 qkv_bias = qkv_bias,
                                 drop = drop,
                                 attn_drop = attn_drop,
                                 drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer = norm_layer
                                 )
            for i in range(depth)])
        
        if downsample is not None:
            self.downsample = downsample(dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        """
        创建mask，用于遮盖不应该注意的位置
        """
        """
        用ceil向上取整计算能够得到多少窗口，然后乘以窗口大小，得到实际的图像大小，
        初始化一个与填充后的图像大小相同的张量，
        """
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        #构建mask
        img_mask = torch.zeros((1, Hp, Wp, 1), device = x.device)
        """
        将特征图进行切片，沿高和宽进行切片，得到3*3的窗口
        """
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        """
        初始化一个计数器，对3*3每个窗口进行遍历，
        在每个窗口填充该窗口的cnt值，
        随后将cnt值+1，确保每个窗口的cnt值不同，从而区分不同的窗口
        """
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        """
        [B*H // window_size[0]*W // window_size, window_size, window_size, 1]->[B*H // window_size[0]*W // window_size, window_size*window_size]
        每层都对应这一个窗口，每个窗口都有唯一且不同的cnt值
        """
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size*self.window_size)
        """
        即[同上个注释]->[B*H // window_size[0]*W // window_size, 1, window_size*window_size] - [B*H // window_size[0]*W // window_size, window_size*window_size, 2]
        随后进行广播操作，得到[B*H // window_size[0]*W // window_size, window_size*window_size, window_size*window_size]
        如果 `attn_mask[i, j, k] == 0`，则表示第 `i` 个窗口中，第 `j` 个像素和第 `k` 个像素的 `cnt` 值相同，它们可以互相注意。
        如果 `attn_mask[i, j, k] != 0`，则表示第 `i` 个窗口中，第 `j` 个像素和第 `k` 个像素的 `cnt` 值不同，它们不应该互相注意。
        """
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        
        return attn_mask
    
    def forward(self, x, H, W):
        attn_mask = attn_mask = self.create_mask(x, H, W)
        for blk in self.blocks:
            #为了获取当前特征图的大小
            blk.H, blk.W = H, W
            #通过梯度检查点机制，减少内存占用
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        #通过下采样层将特征图进行下采样，使得特征图大小减半
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2
        
        return x, H, W

        

class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 window_size = 7,
                 shift_size = 0,
                 mlp_ratio = 4.,
                 qkv_bias = True,
                 drop = 0.,
                 attn_drop = 0.,
                 drop_path = 0.,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm):
        super(SwinTransformerBlock, self).__init__()
        #声明
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        #判断窗口位移的合理性
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim,
                                    window_size = (window_size, window_size),
                                    num_heads = num_heads,
                                    qkv_bias = qkv_bias,
                                    attn_drop = attn_drop,
                                    proj_drop = drop)
        
        #设置随机深度
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim,
                       hidden_features = mlp_hidden_dim,
                       act_layer = act_layer,
                       drop = drop)
    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        """
            移动窗口，移动方向为（-shift_size，-shift_size）
            在（1,2）维度上进行移动,即高和宽方向
        """
        assert L == H*W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        """
        检测tensor的形状是否正确，
        在patchembed中，将图像划分为patch，并返回了x，H，W，
        这里检测输入的x是否与H，W相等
        """
        pad_l = pad_t = 0
        pad_r = (self.window_size - H % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        #划分窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts = (-self.shift_size, -self.shift_size), dims = (1, 2))
        else:
            shifted_x = x
            attn_mask = None
        
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts = (self.shift_size, self.shift_size), dims = (1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H*W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x 

def window_partition(x, window_size):
    """
    将大的特征图分解为多个小的，不重叠的窗口
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    #[(B, H // window_size[0], window_size, W // window_size, window_size, C)]->[(B, H // window_size[0], W // window_size, window_size, window_size, C)]->[(B*H // window_size[0]*W // window_size, window_size, window_size, C)]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口的输出进行重组，得到原图的输出
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    #使用-1自动计算通道数
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):

    #窗口内多头自注意力机制
    
    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias = True,
                 attn_drop = 0.,
                 proj_drop = 0.,):
        super(WindowAttention, self).__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        #即划分出每个头的维度
        head_dim = dim // num_heads
        #缩放因子，防止点积过大导致softmax后梯度消失
        self.scale = head_dim ** -0.5

        #创建偏置表 
        """
        通过nn.Parameter将偏置表注册为可学习的模型参数，
        torch.zeros能够创建一个指定形状的张量，
        第一个维度为(2*window_size[0]-1)*(2*window_size[1]-1)，
        为所有可能得相对位置参数，
        第二个维度为自注意力头数，确保每个头都有自己的偏置表
        """
        self.ralative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads)
        )
        #获取相对位置索引
        """
        coords_h和coords_w分别为行和列的相对位置索引，
        范围为从0到window_size[0]-1和0到window_size[1]-1，
        """
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        """
        torch.mesgrid能够将输入的两个一维张量转换为两个二维张量，
        第一个二维张量（grid_h）的行索引为coords_h，大小为（len(coords_h), len(coords_w)），
        第二个二维张量（grid_w）的列索引为coords_w，大小为（len(coords_h), len(coords_w)），
        torch.stack能够将这两个相同形状的向量沿着一个新的维度进行堆叠（拼接），
        得到一个三维张量，
        此处为（2，window_size[0], window_size[1])，
        """
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        """
        torch.flatten的格式为(input, start_dim, end_dim)
        input为张量，start_dim为起始维度，end_dim为结束维度，
        这里输入coords，将宽和高的张量展平，仅保留行信息和列信息
        例子：展平后coords[0,:]为[0,0,0,...0(window_size[0]-1),1,1,1,...1(window_size[0]-1),...,window_size[1]-1,...,window_size[1]-1(window_size[0]-1)]
        """
        coords_flatten = torch.flatten(coords, 1)

        #利用广播机制获取相对位置索引
        """
        设展评后为N = H * W，
        那么coords的形状为(2, N)，
        那么coords_flatten[:, :, None]为(2, N, 1)，
        那么coords_flatten[:, None, :]为(2, 1, N)，
        相减后得到的relative_coords的形状为(2, N, N)，
        """
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        """
        [2, N, N]->[N, N, 2]
        contiguous()将重排后的张量转换为连续内存
        """
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        """
        dy和dx的值域为-window_size[0]+1到window_size[0]-1和-window_size[1]+1到window_size[1]-1，
        为了转换为索引，需要正值化
        """
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        #通过dy * (2*window_size[0]-1) 确保展平后获得的位置索引不会重叠
        relative_coords[:, :, 0] += 2 * window_size[0] - 1
        """
        对张量的最后一个维度进行求和，得到每个点的相对位置坐标索引，
        将 relative_coords[i, j, 0] 和 relative_coords[i, j, 1] 这两个值相加。
        """
        relative_position_index = relative_coords.sum(-1)
        """
        将relative_position_index张量注册为模块的缓冲区
        缓冲区是不会被梯度更新的，因此不会影响到模型的训练
        """
        self.register_buffer("relative_position_index", relative_position_index)

        #构建qkv层
        
        #将维度*3方便后续切割为q，k，v三个矩阵
        self.qkv = nn.Linear(dim, dim * 3, bias =qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        #对 relative_position_bias_table 进行随机初始化,使用截断正态分布来初始化，std为截断正态分布的标准差
        nn.init.trunc_normal_(self.ralative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim = -1)

    #构建正向传播网络
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        #[B_, N, 3, self.num_heads, C//self.num_heads]->[3, B_, self.num_heads, N, C//self.num_heads]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        #从第一个维度切割q，k，v，分别得到三个矩阵
        q, k, v = qkv.unbind(0)

        #应用缩放因子，保证不会出现梯度消失
        q = q * self.scale
        """
        @表示进行矩阵乘法，
        k.transpose(-2, -1)表示将k矩阵转置，
        """
        attn = (q @ k.transpose(-2, -1))
        """
        外部是为了使用展平后relative_position_index，来从relative_position_bias_table中获取相对位置偏置，
        形状为(N*N, num_heads)，
        随后内部进行重塑，变为（dy，dx，num_heads）格式，对应每个点的相对位置偏置，
        """
        relative_position_bias = self.ralative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)
        """
        (N, N, num_heads)->(num_heads, N, N),
        将张量转换为连续内存
        """
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        """
        [B_, H, N, N] =  [B_, H, N, N] + [1, H, N, N]
        利用广播机制，在relative_position_bias的第一个维度添加一个维度，
        得到[1, H, N, N]，
        然后将attn和relative_position_bias相加，
        得到[B_, H, N, N]
        """
        attn = attn + relative_position_bias.unsqueeze(0)
        
        #应用掩码，确保只会对单一的窗口进行注意力计算
        if mask is not None:
            nw = mask.shape[0]
            """
            B_ // nw表示特征图的数量，nw表示窗口的数量，
            mask.unsqueeze(1).unsqueeze(0)的形状为(nw, N, N)->(nw, 1, N, N)->(1, nw, 1, N, N)
            利用广播机制，进行合并
            """
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        #qkv矩阵乘法的最后一步
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        X = self.proj(x)
        x = self.proj_drop(x)
        return x

class DropPath(nn.Module):
    """
    随机深度，随机丢弃一些层
    """
    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        """
        x.shape[0]获取x的第一个维度，
        (x.ndim - 1)获取x的剩余维度数，
        (1,) * (x.ndim - 1)将剩余维度数的元素组成元组，且全部为1，
        将其拼接，即[B, 1, 1, 1, 1,..., 1]
        """
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        """
        dtype和device属性分别获取张量的类型和所在设备，
        然后生成一个随机张量，范围为0到1，
        并将其与keep_prob相加，
        """
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        """
        进行二值化转换为0或1，
        （向下取整）
        """
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp(nn.Module):
    """
    多层感知机
    用于调整输入输出的维度
    """
    def __init__(self,
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.GELU,
                 drop = 0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
class PatchMerging(nn.Module):
    """
    进行下采样操作，将特征图的尺寸减半
    """
    def __init__(self,
                 dim,
                 norm_layer = nn.LayerNorm):
        super(PatchMerging, self).__init__()
        self.dim = dim
        self.reduction = nn.Linear(4*dim, 2*dim, bias=False)
        self.norm = norm_layer(4*dim)
    
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]

        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4*C)

        x = self.norm(x)
        x = self.reduction(x)
        
        return x


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    """
    构建swin_tiny模型
    """
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model

if __name__ == '__main__':
    model = swin_tiny_patch4_window7_224(num_classes=1000)
    print(model)


