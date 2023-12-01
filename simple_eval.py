import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from dataset import MNISTBoundingBoxDataset, transform, GRID_SIZE, image_width, image_height
from minimodel2 import miniModel

model = miniModel(image_width=image_width, image_height=image_height, image_channels=1, grid_ratio=64, num_bounding_boxes=1, num_classes=10, dropout=0.4)
model.eval()
def paint_bounding_box(box, ax, color="red"):
    x, y, w, h = box
    if(box[2] > 0 and box[3] > 0):
        print(f"box: {box}")
        x, y, w, h = box
        x_abs = (x + i) * cell_width
        y_abs = (y + j) * cell_height
        w_abs = w * image_width
        h_abs = h * image_height
        x_min = x_abs - (w_abs / 2)
        y_min = y_abs - (h_abs / 2)
        # You can now use abs_bounding_box for plotting or other purposes
        rect = patches.Rectangle((x_min, y_min), w_abs, h_abs, 
                                linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

if __name__ == '__main__':
    # Load the dataset
    dataset = MNISTBoundingBoxDataset(root="data", train=True, download=True, transform=transform)

    # Get a sample image, bounding box, and label
    image, target = dataset[60]  # Change 0 to any index to test different samples
    prediction = model(image.unsqueeze(0)).detach()
    bounding_box = target[:4,:,:]
    label = target[4:,:,:]
    # Convert the tensor image back to PIL for display
    pil_img = transforms.ToPILImage()(image).convert("RGB")

    # Create a matplotlib figure
    fig, ax = plt.subplots()
    ax.imshow(pil_img)
    # Add the grid
    cell_width = image_width / GRID_SIZE
    cell_height = image_height / GRID_SIZE

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            rel_target_bounding_box = target[1:5, i, j]  # Extract relative bounding box
            print(f"tarbb_shape: {rel_target_bounding_box.shape}")
            rel_prediction_bounding_box = prediction[0, i, j, 1:5]
            print(f"predbb_shape: {rel_prediction_bounding_box.shape}")
            paint_bounding_box(rel_target_bounding_box, ax, color="red")
            paint_bounding_box(rel_prediction_bounding_box, ax, color="green")

    # Display the image and bounding box
    plt.show()
