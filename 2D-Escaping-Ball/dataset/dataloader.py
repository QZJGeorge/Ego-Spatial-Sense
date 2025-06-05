import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class SenseDataset(Dataset):
    """
    Custom Dataset for Car Following Data.
    Each sample is a concatenation of features from two rows:
      - The first row (starting from row 0)
      - The row located delta_steps ahead.
    The target is taken from the second row.
    """

    def __init__(self, data, input_columns, target_columns, device):
        # Ensure clean indexing and store parameters
        self.data = data.reset_index(drop=True)
        self.device = device
        self.input_columns = input_columns
        self.target_columns = target_columns

    def __len__(self):
        # The last starting index is such that idx + delta_steps is a valid row.
        return len(self.data) - 1

    def __getitem__(self, idx):
        # First sample row (starting at index 0, then increasing by 1)
        first_sample = self.data.iloc[idx]
        # The next sample is delta_steps away from the first sample
        next_sample = self.data.iloc[idx + 1]

        # Extract input features from both rows
        inputs_first = first_sample[self.input_columns].values
        inputs_next = next_sample[self.input_columns].values

        # Concatenate the features from both rows
        inputs = torch.tensor(
            list(inputs_first) + list(inputs_next),
            dtype=torch.float32,
        ).to(self.device)

        # The target is taken from the next sample's target columns
        target = torch.tensor(
            [next_sample[col] for col in self.target_columns], dtype=torch.float32
        ).to(self.device)

        return inputs, target


def get_data(
    file_path,
    input_columns,
    target_columns,
    batch_size=64,
    device="cpu",
):
    """
    Function to prepare a DataLoader for a portion of the dataset.

    Args:
        file_path (str): Path to the CSV file containing the data.
        input_columns (list): List of columns to use as input features.
        target_columns (list): List of columns to use as targets.
        batch_size (int): Batch size for the DataLoader.
        device (str): Device to load the data on ("cpu" or "cuda").
        ratio (float): Ratio of data to load from the beginning (0.0 - 1.0).

    Returns:
        DataLoader: DataLoader for the dataset.
    """
    # Load data from CSV
    data = pd.read_csv(file_path)

    # Create the custom dataset and wrap it in a DataLoader
    dataset = SenseDataset(data, input_columns, target_columns, device)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader
