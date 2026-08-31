import os
import numpy as np
import pickle


# 加载 LIDAR 点云数据
def load_velodyne_bin(file_path):
    """
    从 .bin 文件中加载 LIDAR 点云数据，提取 (x, y, z) 坐标。
    :param file_path: .bin 文件的路径
    :return: 点云的 (x, y, z) 坐标数组
    """
    point_cloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    points = point_cloud[:, :3]  # 提取前 3 列 (x, y, z)
    return points


# 保存点云为 pkl 文件
def save_to_pkl(points, output_file):
    """
    将点云数据 (x, y, z) 保存为 .pkl 文件。
    :param points: 点云的 (x, y, z) 坐标
    :param output_file: 保存的 .pkl 文件路径
    """
    with open(output_file, 'wb') as f:
        pickle.dump(points, f)
    print(f"Saved point cloud to {output_file}")


# 处理一个文件夹中的所有 .bin 文件
def process_bin_folder(input_folder, output_folder):
    """
    将输入文件夹中的所有 .bin 文件转换为 .pkl 文件，并保存到输出文件夹。
    :param input_folder: 包含 .bin 文件的文件夹路径
    :param output_folder: 保存 .pkl 文件的文件夹路径
    """
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有 .bin 文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.bin'):
            lidar_file = os.path.join(input_folder, filename)

            # 加载 LIDAR 点云数据
            points = load_velodyne_bin(lidar_file)

            # 生成对应的 .pkl 文件路径
            output_file = os.path.join(output_folder, filename.replace('.bin', '.pkl'))

            # 保存为 .pkl 文件
            save_to_pkl(points, output_file)


if __name__ == "__main__":
    # 输入文件夹路径 (包含 .bin 文件)
    input_folder = 'data/SemanticKITTI/data_velodyne/velodyne/sequences/00/velodyne'  # 替换为实际的 .bin 文件夹路径

    # 输出文件夹路径 (保存 .pkl 文件)
    output_folder = '/media/hanyong/zgw/chk/DPOcc/kitti_gt_pkl'  # 替换为实际的输出 .pkl 文件夹路径

    # 处理整个文件夹中的 .bin 文件
    process_bin_folder(input_folder, output_folder)