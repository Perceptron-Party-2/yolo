import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MNISTBoundingBoxDataset, transform
#from model import YOLO  # Replace with your YOLO model class
from minimodel2 import miniModel
import wandb
import tqdm
from simple_loss import SimpleLoss


# Configuration parameters
learning_rate = 0.001
batch_size = 64
num_epochs = 10

wandb.login()
wandb.init(
    project="yolo_simple",
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
#model = YOLO(conv_configs=conv_config_yolo).to(device)  # Fill in with appropriate arguments
model = miniModel(image_width=448, image_height=448, image_channels=1, grid_ratio=64, num_bounding_boxes=1, num_classes=10, dropout=0.4)
loss_fn = SimpleLoss().to(device)  # Fill in with appropriate arguments

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (images, targets) in enumerate(tqdm.tqdm(train_loader, desc=f"Epoch {epoch}")):
        print(f"batch: {i}")
        images = images.to(device)
        ground_truth = targets.to(device).permute(0, 2, 3, 1)
        model = model.to(device)
        # Forward pass
        model_output = model(images)  # Model predictions, shape [batch_size, 7, 7, 25]
        loss = loss_fn(model_output, ground_truth)

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