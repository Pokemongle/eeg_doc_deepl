import torch
import numpy as np
from tqdm.notebook import tqdm
import os
import scipy.io
import gc
from load_data import MyData  # self-made
import pandas as pd

def data_cut(dataset, chunk_length, device):
    """
    cut data into chunk_length pieces
    :param dataset:
    :param chunk_length:
    :param device:
    :return:
    """
    # Assuming data is a NumPy array with shape (59, 112008)
    for person in tqdm(range(len(dataset))):  # 12种情况中的1种，其中的所有被试
        # filepath是最子文件夹中每个.mat文件的名字
        # path是包含当前情况的.mat文件的子文件夹
        filename = os.path.join(dataset.path, dataset.file_path[person])  # eg. conditionA\hc\hc1.set.mat
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.mat':
            mat_data = scipy.io.loadmat(filename)
            # Assuming your data is stored under the key 'data' in the .mat file
            data = mat_data['datas']
        elif ext == '.csv':
            data = pd.read_csv(filename).values
        elif ext == '.txt':
            data = np.loadtxt(filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        # data = scipy.io.loadmat(filename)  # 读取该被试的数据为字典
        data = data['datas']  # 键值对读取59*2400*[paras]的数据矩阵
        # 将数据移动到GPU上
        data = torch.from_numpy(data).to(device)
    # Calculate n
    n = data.shape[1] // 2400

    # Reshape the data to (59, 2400, n)
    reshaped_data = data.reshape(59, 2400, n)

    # Convert the reshaped data to a PyTorch tensor
    torch_data = torch.tensor(reshaped_data, dtype=torch.float32)

    # Check the shape of the torch tensor
    print(torch_data.shape)


def generate_eegmap(dataset, matrix_index, exper_dir, condi_dir, device):
    """
    this function will change 59-channel eeg data into 10×11 eeg map
    :param dataset:
    :param matrix_index:
    :param exper_dir:
    :param condi_dir:
    :param device:
    :return:
    """
    for person in tqdm(range(len(dataset))):  # 12种情况中的1种，其中的所有被试
        # filepath是最子文件夹中每个.mat文件的名字
        # path是包含当前情况的.mat文件的子文件夹
        filename = os.path.join(dataset.path, dataset.file_path[person])  # eg. conditionA\hc\hc1.set.mat
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.mat':
            data = scipy.io.loadmat(filename)
        elif ext == '.csv':
            data = pd.read_csv(filename).values
        elif ext == '.txt':
            data = np.loadtxt(filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        # data = scipy.io.loadmat(filename)  # 读取该被试的数据为字典
        # data = scipy.io.loadmat(filename)  # 读取该被试的数据为字典
        data = data['datas']  # 键值对读取59*2400*[paras]的数据矩阵
        # 将数据移动到GPU上
        data = torch.from_numpy(data).to(device)

        # 在GPU上处理数据
        # 创建被试全0map
        data_map_person = []
        print(f"----现在处理：{dataset.file_path[person]}，共{data.shape[2]}段----")
        # 遍历该被试的所有段
        for para in tqdm(range(data.shape[2])):
            # 创建段全0map
            data_map_para = []
            for point in range(data.shape[1]):  # 遍历该段所有数据点
                # 创建单个数据点全0map
                data_map_point = torch.zeros((9, 11), device=device)
                # 遍历59电极数据并赋值给全0map
                for channel in range(data.shape[0]):
                    data_map_point[matrix_index[0][channel]][matrix_index[1][channel]] = data[channel][point][para]
                # 保存单个数据点map到段map列表
                data_map_para.append(data_map_point)
            # 保存 段map 到 被试map
            data_map_person.append(torch.stack(data_map_para))
            # print(len(data_map_person))
            # 清理内存
            del data_map_para
            gc.collect()
            torch.cuda.empty_cache()
        # 保存 被试map 到文件中
        save_path = f"../data/eegmap_direct_new/{exper_dir}/{condi_dir}/{dataset.file_path[person]}.pt"
        torch.save(torch.stack(data_map_person), save_path)


def generate_eegstrip(dataset, matrix_index, exper_dir, condi_dir, device):
    """
    this function will save 59×1 .mat into 59×1 .mat.pt
    :param dataset:
    :param matrix_index:
    :param exper_dir:
    :param condi_dir:
    :param device:
    :return:
    """
    for person in tqdm(range(len(dataset))):  # 12种情况中的1种，其中的所有被试
        # filepath是最子文件夹中每个.mat文件的名字
        # path是包含当前情况的.mat文件的子文件夹
        filename = os.path.join(dataset.path, dataset.file_path[person])  # eg. conditionA\hc\hc1.set.mat
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.mat':
            data = scipy.io.loadmat(filename)
        elif ext == '.csv':
            data = pd.read_csv(filename).values
        elif ext == '.txt':
            data = np.loadtxt(filename)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        # data = scipy.io.loadmat(filename)  # 读取该被试的数据为字典
        # data = scipy.io.loadmat(filename)  # 读取该被试的数据为字典
        data = data['datas']  # 键值对读取59*2400*[paras]的数据矩阵
        # 将数据移动到GPU上
        data = torch.from_numpy(data).to(device)
        row_index = np.arange(59)
        # 在GPU上处理数据
        # 创建被试全0map
        data_map_person = []
        print(f"----现在处理：{dataset.file_path[person]}，共{data.shape[2]}段----")
        # 遍历该被试的所有段
        for para in tqdm(range(data.shape[2])):
            # 创建段全0map
            data_map_para = []
            for point in range(data.shape[1]):  # 遍历该段所有数据点
                # 创建单个数据点全0map
                data_map_point = torch.zeros((59, 1), device=device)
                # 遍历59电极数据并赋值给全0map
                for channel in range(data.shape[0]):
                    data_map_point[row_index[channel]][row_index[0]] = data[channel][point][para]
                # 保存单个数据点map到段map列表
                data_map_para.append(data_map_point)
            # 保存 段map 到 被试map
            data_map_person.append(torch.stack(data_map_para))
            # print(len(data_map_person))
            # 清理内存
            del data_map_para
            gc.collect()
            torch.cuda.empty_cache()
        # 保存 被试map 到文件中
        save_path = f"../data/eegmap_direct_new/rest/{condi_dir}/{dataset.file_path[person]}.pt"
        torch.save(torch.stack(data_map_person), save_path)


def generate_eegmap_chunks(root_dir, exper_dir, condi_dir):
    dataset = MyData(root_dir, exper_dir, condi_dir)
    data_all = []
    for person in tqdm(range(len(dataset))):
        filename = os.path.join(dataset.path, dataset.file_path[person])
        data_map = torch.load(filename)
        print(filename)
        print(data_map.size())

        for i in range(data_map.size(0)):  # 每个被试切割成2400大小的块保存到data_all中
            data_all.append(data_map[i])

    savepath = f"../data/eegmap_chunks_new/{exper_dir}/{condi_dir}/{exper_dir}_{condi_dir}.pt"
    # print(data_all.shape)
    torch.save(torch.stack(data_all), savepath)
    print(len(data_all))
    del data_all
    del dataset
    del data_map
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    import random
    dish_list = ['Guizhou', 'Tony', 'Chongqing']
    max_list = [0,0,0]
    # choose the loop number
    for i in range(7):
        # random shit loop
        for j in range(10):
            random_shit = random.randint(0, 2)
            max_list[random_shit] = max_list[random_shit] + 1
            print(f"{i*10+(j+1)}: {dish_list[random_shit]}")
    print(f"We are going to: {dish_list[max_list.index(max(max_list))]}")