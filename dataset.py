from torch.utils.data.dataset import Dataset
 
class MyCustomDataset(Dataset):
    def __init__(self, filetype, file_count):
        
        
    def __getitem__(self, index):
        # stuff
        return (img, label)
 
    def __len__(self):
        return count