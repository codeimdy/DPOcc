import pickle

# 定义 .pkl 文件的路径
pkl_file_path = '/media/hanyong/zgw/chk/DPOcc/vis/13/sequences/08/predictions/000020.pkl'  # 替换为你的 .pkl 文件路径

# 加载并查看 .pkl 文件的内容
with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

# 输出查看内容
print(data)