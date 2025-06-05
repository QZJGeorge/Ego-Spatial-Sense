import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from dataset.dataloader import get_data
from model.es2 import Es2Model


def train_model(
    model_path,
    log_path,
    model,
    data_loader,
    criterion,
    optimizer_k,
    optimizer_rest,
    scheduler,
    num_epochs=10,
    iteration_step=10,
    update_k=False,
    update_rest=True,
):
    best_loss = float("inf")
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        model.spatial_sense_block.k.requires_grad = update_k

        if epoch % iteration_step == 0 and epoch > 0:
            update_k = not update_k
            update_rest = not update_rest
            print(f"Epoch {epoch + 1}: update_k={update_k}, update_rest={update_rest}")

        with tqdm(
            data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"
        ) as progress_bar:
            for inputs, targets in progress_bar:
                if update_k:
                    optimizer_k.zero_grad()
                elif update_rest:
                    optimizer_rest.zero_grad()

                action = model(inputs)
                loss = criterion(action, targets)
                loss.backward()

                if update_k:
                    optimizer_k.step()
                elif update_rest:
                    optimizer_rest.step()

                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}")
        epoch_losses.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"New best loss: {best_loss:.4f}. Model saved to {model_path}")

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["loss"])
        writer.writerows([[loss] for loss in epoch_losses])
    print(f"Losses saved to {log_path}")


def main():
    parser = argparse.ArgumentParser()

    # General settings
    parser.add_argument("--data_path", type=str, default="dataset/data_1x.csv")
    parser.add_argument("--log_path", type=str, default="loss_es2.csv")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default="pretrained/es2_test.pth")

    # Model configuration
    parser.add_argument("--num_features", type=int, default=360)
    parser.add_argument("--num_actions", type=int, default=2)
    parser.add_argument("--sensing_range", type=float, default=800.0)

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--iteration_step", type=int, default=10)

    args = parser.parse_args()
    print(f"Using device: {args.device}")

    input_columns = [f"scan_{i}" for i in range(args.num_features)]
    target_columns = ["fx", "fy"]

    data_loader = get_data(
        file_path=args.data_path,
        input_columns=input_columns,
        target_columns=target_columns,
        batch_size=args.batch_size,
        device=args.device,
    )

    model = Es2Model(
        num_features=args.num_features,
        num_actions=args.num_actions,
        sensing_range=args.sensing_range,
    ).to(args.device)

    if os.path.exists(args.model_path):
        print(f"Loading model parameters from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path))
    else:
        print("No pre-trained model found. Starting training from scratch.")

    criterion = nn.MSELoss()

    # Split parameter groups
    k_params = [model.spatial_sense_block.k]
    other_params = [
        p for name, p in model.named_parameters() if all(p is not kp for kp in k_params)
    ]

    optimizer_k = optim.Adam(k_params, lr=args.learning_rate * 0.2)
    optimizer_rest = optim.Adam(other_params, lr=args.learning_rate)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_rest, mode="min", factor=0.5, patience=10
    )

    train_model(
        model_path=args.model_path,
        log_path=args.log_path,
        model=model,
        data_loader=data_loader,
        criterion=criterion,
        optimizer_k=optimizer_k,
        optimizer_rest=optimizer_rest,
        scheduler=scheduler,
        num_epochs=args.num_epochs,
        iteration_step=args.iteration_step,
        update_k=False,
        update_rest=True,
    )


if __name__ == "__main__":
    main()
