from torch.utils.data import Dataset


class ComposeDataset(Dataset):
    def __init__(self, datasets):
        super(ComposeDataset, self).__init__()
        self.datasets = datasets
        self.lengths = [ds.__len__() for ds in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                return self.datasets[i].__getitem__(idx)
            idx -= length
        raise IndexError("Index out of range")
