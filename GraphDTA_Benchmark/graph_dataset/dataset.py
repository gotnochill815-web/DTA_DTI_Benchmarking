"""
PyTorch Geometric dataset for Drug-Target Affinity prediction.

Each sample contains:
    - Molecular graph
    - Encoded protein sequence
    - Affinity label (pKd)
"""

import os
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class GraphDTADataset(InMemoryDataset):
    """
    Dataset class for GraphDTA.
    """

    def __init__(
        self,
        root,
        dataset_name,
        smiles=None,
        proteins=None,
        labels=None,
        smile_graph=None,
        transform=None,
        pre_transform=None,
    ):
        self.dataset_name = dataset_name

        # Raw data is required only when creating the dataset from scratch
        self.smiles = smiles
        self.proteins = proteins
        self.labels = labels
        self.smile_graph = smile_graph

        super().__init__(root, transform, pre_transform)

        if os.path.isfile(self.processed_paths[0]):
            print(f"Loading processed dataset: {self.processed_paths[0]}")
            self.data, self.slices = torch.load(
                self.processed_paths[0],
                weights_only=False,
            )
        else:
            # Validate that raw data is provided when processed file doesn't exist
            assert smiles is not None, "SMILES strings must be provided"
            assert proteins is not None, "Protein sequences must be provided"
            assert labels is not None, "Labels must be provided"
            assert smile_graph is not None, "smile_graph must be provided"

            print("Processed dataset not found.")
            print("Creating graphs...")
            self.process()
            self.data, self.slices = torch.load(
                self.processed_paths[0],
                weights_only=False,
            )

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}.pt"]

    def download(self):
        pass

    def process(self):

        assert (
            len(self.smiles)
            == len(self.proteins)
            == len(self.labels)
        ), "SMILES, proteins, and labels must have the same length"

        data_list = []

        total = len(self.smiles)

        for idx in range(total):

            if (idx + 1) % 500 == 0 or idx == total - 1:
                print(f"Processing {idx+1}/{total}")

            smile = self.smiles[idx]
            protein = self.proteins[idx]
            label = self.labels[idx]

            num_atoms, node_features, edge_index = self.smile_graph[smile]

            graph = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                y=torch.tensor([label], dtype=torch.float),
            )

            graph.target = torch.tensor([protein], dtype=torch.long)

            graph.c_size = torch.tensor([num_atoms], dtype=torch.long)

            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [
                graph
                for graph in data_list
                if self.pre_filter(graph)
            ]

        if self.pre_transform is not None:
            data_list = [
                self.pre_transform(graph)
                for graph in data_list
            ]

        data, slices = self.collate(data_list)

        torch.save(
            (data, slices),
            self.processed_paths[0],
        )

        print("Dataset saved to:")
        print(self.processed_paths[0])