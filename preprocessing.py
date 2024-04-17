import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import torch

# =========================== 导入 Monai 库 ===========================================
from monai.apps import DecathlonDataset   
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
)
# print_config()


# ======================= 准备工作 ===========================================================
set_determinism(seed=0)

if os.path.exists('/mnt/g/DATASETS/'):
    data_dir = '/mnt/g/DATASETS/BraTS21/BraTS2021_Training_Data'
    train_txt = os.path.join(data_dir, 'train_list.txt')
else:
    data_dir = 'G:\\DATASETS\\BraTS21\\BraTS2021_Training_Data'
    train_txt = os.path.join(data_dir, 'train_list.txt')


with open(train_txt, 'r') as f:
    paths = [os.path.join(data_dir, line.strip()) for line in f]   # 获取文件夹地址

# file_paths = 
# 获取数据集的类别名
class_name = sorted(x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x)))


"""
将Brats数据集中的标签转换为多通道形式。
Brats数据集中的标签有三个值：
1 : 水肿
2 : 增强肿瘤
3 : 坏死非增强肿瘤核心
、2和3，分别表示、和。
这个类将这三个标签转换为三个通道:
- 肿瘤核心（TC） (2+3)
- 全肿瘤（WT）   (1+2+3)
- 增强肿瘤（ET）   (2)

"""
# ==========================================================================


# ============================== 数据预处理 =================================

## ==================== 通道定义 ===========================================
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(torch.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(torch.logical_or(torch.logical_or(d[key] == 2, d[key] == 3), d[key] == 1))
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = torch.stack(result, axis=0).float()
        return d
## =========================================================================

## ============================ 数据增强 ====================================
train_transform = Compose(
    [
        # load 4 Nifti images and stack them together
        LoadImaged(keys=["image", "label"]),
        # make sure the channel dimension is in the first dimension
        EnsureChannelFirstd(keys="image"),
        # convert the images to the correct data types
        EnsureTyped(keys=["image", "label"]),
        # convert the labels to multi-channel format
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        # reorient the images to RAS+ orientation
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # 重采样，使图片具有各项同性分辨率
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        # crop the images to have the same size
        RandSpatialCropd(keys=["image", "label"], roi_size=[224, 224, 144], random_size=False),
        # 沿着空间轴，随机翻转
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        # 对非零像素值(nonzero)进行归一化处理
        # 归一化可以提高图像的质量和稳定性
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # 随机调整图像的亮度,调整强度为原来的10%，100%概率调整
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        # 随机强度偏移是通过在图像的每个像素上随机采样一个偏移量，然后将该像素的强度值加上这个偏移量来实现的。
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ]
)
val_transform = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        # make sure the channel dimension is in the first dimension
        EnsureChannelFirstd(keys="image"),
        # convert the images to the correct data types
        EnsureTyped(keys=["image", "label"]),
        # convert the labels to multi-channel format
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        # reorient the images to RAS+ orientation
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        # resample the images to have isotropic resolution
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        # normalize the intensity of the images
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)
## ==========================================================================


# directory = os.environ.get("MONAI_DATA_DIRECTORY")
if os.path.exists('/mnt/g/DATASETS/'):
    directory = '/mnt/g/DATASETS/'
else:
    directory = 'G:\\DATASETS\\'
    
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# =================== 通过Monai构建数据集与迭代器 ====================================
train_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=train_transform,
    section="training",
    download=False,
    cache_rate=0.0,
    num_workers=4, 
)

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=4)

val_ds = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    transform=val_transform,
    section="validation",
    download=False,
    cache_rate=0.0,
    num_workers=4,
)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)