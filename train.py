import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MNISTBoundingBoxDataset, transform
from model import YOLO  # Replace with your YOLO model class
from loss import YOLOLoss  # Replace with your YOLO loss class
import wandb
import tqdm
import conv_config_yolo
# Configuration parameters
learning_rate = 0.001
batch_size = 64
num_epochs = 1

wandb.login()
wandb.init(
    project="yolo",
    config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    },
    group="perceptrongang",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
train_dataset = MNISTBoundingBoxDataset(root="data", train=True, download=True, transform=transform)  # Fill in with appropriate arguments
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model and loss function
model = YOLO(conv_configs=conv_config_yolo).to(device)  # Fill in with appropriate arguments
criterion = YOLOLoss().to(device)  # Fill in with appropriate arguments

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")):
        print(f"batch: {i}")
        # print(f"images.shape: {images.shape}")
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss every batch
        print(f"loss: {loss}")
        wandb.log({"loss": loss})
        
        if (i + 1) % 10 == 0:  # Print every 10th batch
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), f"yolo_{epoch}.pth")
print("Training completed.")

wandb.finish()