from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
 
class FMData(Dataset):
    def __init__(self, file):
        self.file = file
        
    def __getitem__(self, index):
        # stuff
        return (torch.LongTensor(self.file[:, 1:]), torch.LongTensor(self.file[:, 0]))
 
    def __len__(self):
        return self.file.shape[0]