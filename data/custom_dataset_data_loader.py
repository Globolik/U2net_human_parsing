import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import Deep_fashion
    dataset = Deep_fashion()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset
def CreateDataset_original(opt):
    dataset = None
    from data.aligned_dataset import Deep_fashion_original_dataset
    dataset = Deep_fashion_original_dataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset
def CreateDataset_11_labels(opt):
    dataset = None
    from data.aligned_dataset import Deep_fashion_11_labels
    dataset = Deep_fashion_11_labels()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset
def CreateDataset_9_labels(opt):
    dataset = None
    from data.aligned_dataset import Deep_fashion_9_labels
    dataset = Deep_fashion_9_labels()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset
def CreateDataset_19_imat_labels(opt):
    dataset = None
    from data.aligned_dataset import Imat_19
    dataset = Imat_19()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset
def CreateDataset_19_imat_labels_QAT(opt):
    dataset = None
    from data.aligned_dataset import Imat_19_QAT
    dataset = Imat_19_QAT()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, train=True):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(self.dataset,
                                 not opt.serial_batches, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
class CustomDatasetDataLoader_original_datastet(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, train=True):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset_original(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(self.dataset,
                                 not opt.serial_batches, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
class CustomDatasetDataLoader_11(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, train=True):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset_11_labels(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(self.dataset,
                                 not opt.serial_batches, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
class CustomDatasetDataLoader_9(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, train=True):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset_9_labels(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(self.dataset,
                                 not opt.serial_batches, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
class CustomDatasetDataLoader_19_imat(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, train=True):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset_19_imat_labels(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(self.dataset,
                                 opt.shuffle, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
class CustomDatasetDataLoader_19_imat_QAT(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, train=True):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset_19_imat_labels_QAT(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            sampler=data_sampler(self.dataset,
                                 opt.shuffle, opt.distributed),
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)


    if shuffle:
        return torch.utils.data.RandomSampler(dataset)

    else:
        return torch.utils.data.SequentialSampler(dataset)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


class CustomTestDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            num_workers=int(opt.nThreads),
            pin_memory=True)

    def get_loader(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
