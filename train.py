import torch
from Networks import UNet_3D



max_epochs = 150  # TODO : 修改epoch
val_interval = 1
VAL_AMP = True

# standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
device = torch.device("cuda:0")
# model = SegResNet(
#     blocks_down=[1, 2, 2, 4],
#     blocks_up=[1, 1, 1],
#     init_filters=16,
#     in_channels=4,
#     out_channels=3,
#     dropout_prob=0.2,
# ).to(device)
model = UNet_3D(in_channels=4,num_classes=3).to(device)  # TODO：根据类别数修改

# print(model.parameters())
loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)

## =================优化器=================================================
# optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)

## ================学习率调度器==========================================
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

# ================评估指标=================================================
dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
