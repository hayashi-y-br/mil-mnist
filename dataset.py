import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MyDataset(Dataset):
    def __init__(
            self,
            root='./data',
            train=True,
            train_size=800,
            test_size=200,
            train_seed=0,
            test_seed=0,
            bag_size=16,
            blank_ratio_low=25,
            blank_ratio_high=75,
            target_numbers=[0, 1, 2]):
        self.root = root
        self.train = train
        self.num_samples = train_size if self.train else test_size
        self.seed = train_seed if self.train else test_seed
        self.bag_size = bag_size
        self.blank_ratio_low = blank_ratio_low
        self.blank_ratio_high = blank_ratio_high
        self.target_numbers = target_numbers
        self.rng = np.random.default_rng(self.seed)
        self.dataset = datasets.MNIST(
            root=self.root,
            train=self.train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        self.class_indices = [[] for _ in range(10)]
        for i, c in enumerate(self.dataset.targets):
            self.class_indices[c].append(i)
        for i in range(10):
            self.class_indices[i] = np.array(self.class_indices[i])
        self.X_indices = []
        self.y = []
        for i in range(self.num_samples):
            label = i % 4
            num_blank = self.bag_size
            num_plus = 0
            num_minus = 0
            if label != 0:
                num_blank = self.rng.integers(
                    self.bag_size * blank_ratio_low // 100,
                    min(self.bag_size - 2, (self.bag_size * blank_ratio_high + 99) // 100),
                    endpoint=True
                )
                if label == 1:
                    num_plus = self.bag_size - num_blank
                elif label == 2:
                    num_minus = self.bag_size - num_blank
                else:
                    num_plus = self.rng.integers(1, self.bag_size - num_blank)
                    num_minus = self.bag_size - num_blank - num_plus
            self.X_indices.append(np.concatenate([
                self.class_indices[self.target_numbers[2]][self.rng.integers(self.class_indices[self.target_numbers[2]].size, size=num_minus)],
                self.class_indices[self.target_numbers[0]][self.rng.integers(self.class_indices[self.target_numbers[0]].size, size=num_blank)],
                self.class_indices[self.target_numbers[1]][self.rng.integers(self.class_indices[self.target_numbers[1]].size, size=num_plus)]
            ]))
            self.y.append(label)
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.stack([self.dataset[i][0] for i in self.X_indices[idx]]), self.y[idx]


if __name__ == '__main__':
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    dataset = MyDataset(train=False, test_size=40)
    for i, (X, y) in enumerate(dataset):
        img = make_grid(X, nrow=4, padding=0)[0]
        img = img * 0.5 + 0.5
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        ax.imshow(img, cmap='gray')
        fig.savefig(f'./data/bag/bag_{y}_{i // 4}')
        plt.close(fig)