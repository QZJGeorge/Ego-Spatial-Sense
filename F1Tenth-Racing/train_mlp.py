import csv
import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset.dataloader import get_data
from model.mlp import MLPModel


def train_model(
    model_path,
    log_path,
    model,
    data_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs=10,
):
    """
    Train the model using the provided data loader, loss function, optimizer, and scheduler.

    Args:
        model_path (str): Path to save the best model.
        model (torch.nn.Module): The neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader for training data.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int, optional): Number of training epochs. Defaults to 10.
    """
    best_loss = float("inf")

    # List to store losses per epoch
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Progress bar for training
        with tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ) as progress_bar:
            for inputs, targets in progress_bar:
                optimizer.zero_grad()

                # Forward pass
                actions = model(inputs)
                steer_loss = criterion(actions[:, 0], targets[:, 0])
                speed_loss = criterion(actions[:, 1], targets[:, 1])
                loss = steer_loss + speed_loss * 0.05

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                progress_bar.set_postfix(loss=loss.item())

        # Compute average loss and update scheduler
        avg_loss = total_loss / len(data_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}, ")

        # Record epoch losses
        epoch_losses.append(avg_loss)

        scheduler.step(avg_loss)

        # Save model if new best loss is found
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"New best loss: {best_loss:.4f}. Model saved to {model_path}")

    # Save losses to CSV after training
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss"])  # Pass a list of column headers
        writer.writerows(
            [[loss] for loss in epoch_losses]
        )  # Each loss as a one-item list
    print(f"Losses saved to {log_path}")


def main():
    # Argument parsing
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--data_path", type=str, default="dataset/austin.csv")
    parser.add_argument("--log_path", type=str, default="loss_mlp.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="pretrained/mlp.pth")

    # Model configuration
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=10.0)

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=500)

    args = parser.parse_args()
    print(f"Using device: {args.device}")

    # Define input and target columns
    input_columns = [f"lidar_{i}" for i in range(360)]
    target_columns = ["steer", "desired_speed"]

    # Load training data
    data_loader = get_data(
        file_path=args.data_path,
        input_columns=input_columns,
        target_columns=target_columns,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = MLPModel(
        num_features=args.num_features,
        num_actions=args.num_actions,
        sensing_range=args.sensing_range,
    ).to(args.device)

    print(f"MLP model initialized")

    # Load pretrained model if available
    if os.path.exists(args.model_path):
        print(f"Loading model parameters from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
    else:
        print("No pre-trained model found. Starting training from scratch.")

    # Define loss function, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    # Train the model
    train_model(
        args.model_path,
        args.log_path,
        model,
        data_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    main()
