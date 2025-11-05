from random import Random

from torch.utils.data import DataLoader
import torch.distributed as dist


class Partition:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        """Given index, get the data according to the partitioned index"""
        # BEGIN ASSIGN5_1_1
        return self.data[self.index[index]]
        # END ASSIGN5_1_1


class DataPartitioner:
    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        """ Create indices for different partitions
        1. Create indices and use `rng` to shuffle indices
        2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        """
        # BEGIN ASSIGN5_1_1
        indices = list(range(len(data)))
        # 1. Create indices and use `rng` to shuffle indices
        rng.shuffle(indices)
        # 2. Create different partitions of indices according to `sizes` and store in `self.partitions`
        partition_sizes = [int(s * len(data)) for s in sizes]
        partition_sizes[-1] = len(data) - sum(partition_sizes[:-1])

        # FIX: Use cumulative offsets to avoid overlapping partitions
        start_idx = 0
        for size in partition_sizes:
            self.partitions.append(
                Partition(data, indices[start_idx : start_idx + size])
            )
            start_idx += size
        # END ASSIGN5_1_1

    def use(self, partition):
        """Return a simple dataset class `Partiton` by original data and partitioned indices

        Just one line of code. Think it simply.
        """
        # BEGIN ASSIGN5_1_1
        # FIX: Return the partition object directly (it's already a Partition)
        return self.partitions[partition]
        # END ASSIGN5_1_1


def partition_dataset(rank, world_size, dataset, batch_size=128, collate_fn=None):
    """Partitioning training dataset of the Machine Translation

    Returns:
        DataLoader: partitioned dataloader

    Hint:
    1. Calculate the partitioned batch size
    2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    """
    # BEGIN ASSIGN5_1
    # 1. Calculate the partitioned batch size
    partitioned_batch_size = batch_size // world_size
    # 2. Create a partitioner class `DataPartitioner` with dataset and the list of partitioned sizes
    # FIX: Use fractional sizes (summing to 1.0) so DataPartitioner splits the dataset correctly
    sizes = [1.0 / world_size] * world_size
    partitioner = DataPartitioner(dataset, sizes)
    # 3. Get the current partition dataset given `rank`, use the `use` function in DataPartitioner
    partition = partitioner.use(rank)
    # 4. Wrap the dataset with `DataLoader`, remember to customize the `collate_fn`
    dataloader = DataLoader(
        partition, batch_size=partitioned_batch_size, collate_fn=collate_fn
    )
    return dataloader
    # END ASSIGN5_1
