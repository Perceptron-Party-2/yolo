from dataset import MNISTBoundingBoxDataset, transform
from minimodel2 import miniModel
import torch

# Load the model
model = miniModel(image_width=448, image_height=448, image_channels=1, grid_ratio=64, num_bounding_boxes=2, num_classes=10, dropout=0.3)

# Load the weights
model.load_state_dict(torch.load("yolo_0.pth"))

# Set the model in evaluation mode
model.eval()

# Load the dataset
dataset = MNISTBoundingBoxDataset(root="data", train=True, download=True, transform=transform)

# Get a sample image, bounding box, and label
image, target = dataset[10]  # Change 0 to any index to test different samples

print(f"target.shape: {target.shape}")
# print(f"target: {target}")
bounding_box = target[:,:,:4]
label = target[:,:,4]
# Print the label
#print(f"Label: {label}")
#print(f"Bounding box: {bounding_box}")

# Add a batch dimension
image = image.unsqueeze(0)

print(f"image.shape: {image.shape}")

# Get the predicted bounding box
predicted_bounding_box = model(image)
print(predicted_bounding_box)
#print(f"Predicted bounding box shape: {predicted_bounding_box.shape}")

