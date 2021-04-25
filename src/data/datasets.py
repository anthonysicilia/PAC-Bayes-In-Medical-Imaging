import torch
import pandas
import os

from PIL import Image
from torchvision.transforms import Resize, ToTensor

TASK1_IMG_DIR = 'ISIC2018/ISIC2018_Task1-2_Training_Input/'
TASK1_LABEL_DIR = 'ISIC2018/ISIC2018_Task1_Training_GroundTruth'
TASK3_IMG_DIR = 'ISIC2018/ISIC2018_Task3_Training_Input'
TASK3_LABEL_PATH = 'ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv'
TASK3_LABELS = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

class ISICChallengeSet(torch.utils.data.Dataset):

    def __init__(self, path_file, task=1):
        lines = list(open(path_file, 'r').readlines())
        self.paths = [line.strip() for line in lines]
        self.task = task

        if self.task == 3:
            df = pandas.read_csv(TASK3_LABEL_PATH)
            # double check unique label per row
            assert len(df) == df[TASK3_LABELS].sum().sum()
            df['label'] = df[TASK3_LABELS].idxmax(axis=1
                ).map({k : i for i,k in enumerate(TASK3_LABELS)})
            self.labels = {k : l for k,l in zip(df['image'], df['label'])}
            self.img_dir = TASK3_IMG_DIR
        elif self.task == 1:
            self.img_dir = TASK1_IMG_DIR
            self.label_dir = TASK1_LABEL_DIR
            self.get_label_path = lambda img: f'{img}_segmentation.png'
        else:
            raise NotImplementedError(f'Unrecognixed task {self.task}')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(os.path.join(self.img_dir, path))
        img = ToTensor()(Resize((224, 224))(img))
        path = path.split('.')[0]
        if self.task == 3:
            label = self.labels[path]
        else:
            lp = self.get_label_path(path)
            label = Image.open(os.path.join(self.label_dir, lp))
            label = ToTensor()(Resize((224, 224))(label))
        return {'img' : img, 'label' : label}