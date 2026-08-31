import os
import numpy as np
import pickle

# 定义源文件夹路径（包含 .npy 文件的文件夹）
source_folder_path = '/media/hanyong/zgw/chk/DPOcc/data/SemanticKITTI/labels/09'  # 替换为你的 .npy 文件夹路径

# 定义目标文件夹路径（保存 .pkl 文件的文件夹）
target_folder_path = '/media/hanyong/zgw/chk/DPOcc/data/SemanticKITTI/labels/09/pkl'  # 替换为保存 .pkl 文件的文件夹路径

# 如果目标文件夹不存在，创建文件夹
if not os.path.exists(target_folder_path):
    os.makedirs(target_folder_path)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder_path):
    # 检查文件是否为 .npy 文件
    if filename.endswith('.npy'):
        npy_file_path = os.path.join(source_folder_path, filename)  # 获取完整的 .npy 文件路径

        # 读取 .npy 文件
        data = np.load(npy_file_path)

        # 定义对应的 .pkl 文件路径（保持相同的文件名但后缀改为 .pkl），保存到目标文件夹
        pkl_file_path = os.path.join(target_folder_path, filename.replace('.npy', '.pkl'))

        # 将数据保存为 .pkl 文件
        with open(pkl_file_path, 'wb') as f:
            pickle.dump(data, f)

        print(f"成功将 {npy_file_path} 转换并保存为 {pkl_file_path}")

print("所有文件已转换并保存到目标文件夹。")