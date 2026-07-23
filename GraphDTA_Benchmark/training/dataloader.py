import torch
from torch_geometric.loader import DataLoader

from graph_dataset.dataset import GraphDTADataset


def get_dataloaders(
    data_root,
    batch_size,
    dataset_seed=32,
    num_workers=2,
    shuffle_train=True,
):

    train_dataset = GraphDTADataset(
        root=data_root,
        dataset_name=f"train_{dataset_seed}",
    )

    val_dataset = GraphDTADataset(
        root=data_root,
        dataset_name=f"val_{dataset_seed}",
    )

    test_dataset = GraphDTADataset(
        root=data_root,
        dataset_name=f"test_{dataset_seed}",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader