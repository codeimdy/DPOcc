# import numpy as np
#
# # 定义 .npy 文件的路径
# npy_file_path = '/media/hanyong/zgw/chk/SparseOcc-main/nuscenes_test_pkl/3b2ee26cb8484f77895bc336663df502/2b948e3a2e934c3998bb0185667f808f/pred_f.npy'  # 替换为你的 .npy 文件路径
#
# # 加载 .npy 文件
# data = np.load(npy_file_path)
#
# # 查看数据
# print(data)
#
# # 如果想查看数据的形状、数据类型等信息，可以使用以下命令
# print(f"数据形状: {data.shape}")
# print(f"数据类型: {data.dtype}")

import numpy as np

data = np.fromfile('/media/hanyong/zgw/chk/DPOcc/data/SemanticKITTI/dataset/sequences/00/velodyne/000007.bin', dtype=np.uint8)
print(data)
print(f"数据形状: {data.shape}")
