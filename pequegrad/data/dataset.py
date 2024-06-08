from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
