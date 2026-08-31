import torch
file_path = '/media/hanyong/zgw/chk/DPOcc/work_dirs/best_semkitti_SSC_mIoU_epoch_30.pth'
model = torch.load(file_path, map_location='cpu')
all = 0
for key in list(model['state_dict'].keys()):
    all += model['state_dict'][key].nelement()
print(all)

# smaller 63374123
# v4 69140395
