from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
 
class FMData(Dataset):
    def __init__(self, file):
        self.file = torch.LongTensor(file)
        
    def __getitem__(self, index):
        # stuff
        # return (torch.LongTensor(self.file[index, 2:]), 
        #         torch.LongTensor(self.file[index, 0]), 
        #         torch.LongTensor(self.file[index, 1]))
        return self.file[index]
 
    def __len__(self):
        return self.file.shape[0]