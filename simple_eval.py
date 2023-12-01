import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from dataset import (
    MNISTBoundingBoxDataset,
    transform,
    GRID_SIZE,
    image_width,
    image_height,
)
from minimodel2 import miniModel
from io import BytesIO
import base64

model = miniModel(
    image_width=image_width,
    image_height=image_height,
    image_channels=1,
    grid_ratio=64,
    num_bounding_boxes=1,
    num_classes=10,
    dropout=0.4,
)
model.load_state_dict(torch.load("yolo_2_299.pth"))
model.eval()


def paint_bounding_box(box, digit, confidence, i,j,cell_width,cell_height, ax, color="red"):
    x, y, w, h = box
    if box[2] > 0 and box[3] > 0:
        x, y, w, h = box
        x_abs = (x + i) * cell_width
        y_abs = (y + j) * cell_height
        w_abs = w * image_width
        h_abs = h * image_height
        x_min = x_abs - (w_abs / 2)
        y_min = y_abs - (h_abs / 2)
        # You can now use abs_bounding_box for plotting or other purposes
        rect = patches.Rectangle(
            (x_min, y_min), w_abs, h_abs, linewidth=1, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.annotate(f"{digit} ({confidence})", (x_min, y_min), color=color, weight='bold', fontsize=10, ha='left', va='bottom')

def create_image(image, target, prediction, base64_out=False):
    print(f"image.shape: {image.shape}")
    assert image.shape == (1, 448, 448)
    print(f"prediction.shape: {prediction.shape}")
    assert prediction.shape == (GRID_SIZE, GRID_SIZE, 15)
     # Convert the tensor image back to PIL for display
    pil_img = transforms.ToPILImage()(image).convert("RGB")

    # Create a matplotlib figure
    _, ax = plt.subplots()
    ax.imshow(pil_img)
    # Add the grid
    cell_width = image_width / GRID_SIZE
    cell_height = image_height / GRID_SIZE

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            target_confidence = torch.zeros(1,1,1)
            if target != None:
                target_confidence = target[0, i, j]
                target_bounding_box = target[1:5, i, j]  # Extract relative bounding box
                target_label = target[5:, i, j]
                target_digit = torch.argmax(
                    target_label
                )  # argmax returns the index of the maximum value
            prediction_confidence = prediction[i, j, 0]
            prediction_bounding_box = prediction[i, j, 1:5]
            prediction_label = prediction[i, j, 5:]
            prediction_digit = torch.argmax(prediction_label)
            CONFIDENCE_THRESHOLD = 0.2
            if target_confidence > CONFIDENCE_THRESHOLD:
                paint_bounding_box(
                    target_bounding_box,
                    # digit is a tensor, so we need to convert it to a Python int
                    target_digit.item(),
                    # round confidence to 2 decimal places
                    round(target_confidence.item(),2),
                    i,
                    j,
                    cell_width,
                    cell_height,
                    ax,
                    color="red",
                )
            if prediction_confidence > CONFIDENCE_THRESHOLD:
                paint_bounding_box(
                    prediction_bounding_box,
                    prediction_digit.item(),
                    round(prediction_confidence.item(),2),
                    i,
                    j,
                    cell_width,
                    cell_height,
                    ax,
                    color="green",
                )

    if base64_out:
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    
    return plt


if __name__ == "__main__":
    # Load the dataset
    dataset = MNISTBoundingBoxDataset(
        root="data", train=True, download=True, transform=transform
    )

    # Get a sample image, bounding box, and label
    image, target = dataset[15]  # Change 0 to any index to test different samples
    print(f"image.shape: {image.shape}")
    prediction = model(image.unsqueeze(0)).detach().squeeze(0)

    plt = create_image(image, target, prediction)
    print(create_image(image, target, prediction, base64_out=True))

    # Display the image and bounding box
    plt.show()
