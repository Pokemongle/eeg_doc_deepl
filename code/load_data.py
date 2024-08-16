import scipy.signal
import torch
import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision.transforms import Normalize, ToTensor
from tqdm import tqdm_notebook as tqdm
from PIL import Image
import scipy.io
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np



class MyData(Dataset):
    """
    @ How to get data
    Use MyData[idx] to get data of each file
    Use len(MyData) to check the length
    @ args
    root_dir: train or test
    label_dir:
        train dataset: ECG or Peak
        test dataset: exercise or rest, then ECG or Peak
    """

    def __init__(self, root_dir, experiment_dir, condition_dir):
        self.root_dir = root_dir
        self.experiment_dir = experiment_dir
        self.condition_dir = condition_dir
        # path for dir containing txt files
        self.path = os.path.join(self.root_dir, self.experiment_dir, self.condition_dir)
        # name of each txt file
        self.file_path = os.listdir(self.path)

    def __getitem__(self, idx):
        """
        get the No.idx Image object of the label directory and its label
        called when index is used: MyData[index]
        e.g. ants_dataset[0]
        :param idx: No.idx file in the label_dir
        :return: the No.idx image and its label
        """
        # get the name of one file in the file_path dir
        file_name = self.file_path[idx]
        # get the path of the file
        file_item_path = os.path.join(self.root_dir, self.experiment_dir, self.condition_dir, file_name)
        # --get data from the path
        with open(file_item_path, 'r') as file:
            data = [float(value) for value in file.read().split()]
        return data

    def __len__(self):
        """
        get the length of the dataset
        :return:
        """
        return len(self.file_path)


if __name__ == "__main__":
    root_dir = "../eegdata"
    # experimental dir: rest, conditionA, conditionB, conditionC
    exper_dir = "rest"
    # condition_dir: hc, mcs, uws
    condi_dir = "hc"
    rest_hc_dataset = MyData(root_dir, exper_dir, condi_dir)
    # filepath是最子文件夹中每个.mat文件的名字
    # path是包含当前情况的.mat文件的子文件夹
    # length = len(rest_hc_dataset)
    mat_a = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ])
    mat_index = np.array([
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ])
    mat_map = np.array([])
    keep_map = np.zeros((2, 2, 1))
    for j in range(mat_a.shape[0]):
        map = np.zeros((2, 2))
        print(map)
        for i in range(mat_a.shape[1]):
            map[mat_index[0][i]][mat_index[1][i]] = mat_a[j][i]
        print(map)
        keep_map = np.concatenate((keep_map, map), axis=2)

