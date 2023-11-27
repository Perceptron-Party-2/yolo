from dataset import MNISTBoundingBoxDataset, transform
from model import YOLO

# Load the model
model = YOLO()

# Load the weights
# model.load_state_dict(torch.load("yolo.pth"))

# Set the model in evaluation mode
model.eval()

# Load the dataset
dataset = MNISTBoundingBoxDataset(root="data", train=True, download=True, transform=transform)

# Get a sample image, bounding box, and label
image, bounding_box, label = dataset[10]  # Change 0 to any index to test different samples

# Print the label
print(f"Label: {label}")

# Add a batch dimension
image = image.unsqueeze(0)

print(f"image.shape: {image.shape}")

# Get the predicted bounding box
predicted_bounding_box = model(image)

# Print the shape of the output
print(f"Predicted bounding box shape: {predicted_bounding_box.shape}")

